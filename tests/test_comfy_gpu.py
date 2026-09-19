"""Opt-in end-to-end tests through the actual ComfyUI node entry points."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import av

from dlss_engine.core.paths import FFMPEG


@unittest.skipUnless(
    os.environ.get("DLSS_RUN_COMFY_GPU_TESTS") == "1",
    "Set DLSS_RUN_COMFY_GPU_TESTS=1 and DLSS_COMFYUI_PATH for node GPU tests",
)
class ComfyNodeGPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        comfy_path = Path(os.environ["DLSS_COMFYUI_PATH"]).resolve()
        if not (comfy_path / "folder_paths.py").is_file():
            raise RuntimeError("DLSS_COMFYUI_PATH must point to ComfyUI")
        sys.path.insert(0, str(comfy_path))
        import folder_paths
        import torch

        cls.torch = torch
        cls.folder_paths = folder_paths
        cls.old_temp = folder_paths.get_temp_directory()
        cls.temp = tempfile.TemporaryDirectory(prefix="dlss-comfy-gpu-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.addClassCleanup(folder_paths.set_temp_directory, cls.old_temp)
        folder_paths.set_temp_directory(cls.temp.name)
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "dlss_test_nodes", root / "__init__.py",
            submodule_search_locations=[str(root)],
        )
        cls.nodes = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.nodes
        spec.loader.exec_module(cls.nodes)
        cls.source = Path(cls.temp.name) / "input.mp4"
        subprocess.run([
            str(FFMPEG), "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=0.4",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=0.4",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-shortest", str(cls.source),
        ], check=True, timeout=30)

    def check_video(self, result, dimensions, frame_count, fps):
        path = result.result[0].get_stream_source()
        with av.open(path) as video:
            self.assertEqual(len(video.streams.audio), 1)
            stream = video.streams.video[0]
            self.assertEqual(stream.average_rate, fps)
            frames = list(video.decode(stream))
            self.assertEqual(len(frames), frame_count)
            self.assertEqual((frames[0].width, frames[0].height), dimensions)
        return json.loads(result.result[1])

    def image_batch(self, frames=5, height=360, width=640):
        torch = self.torch
        x = torch.linspace(0, 1, width).reshape(1, 1, width, 1)
        y = torch.linspace(0, 1, height).reshape(1, height, 1, 1)
        base = torch.cat((x.expand(1, height, width, 1), y.expand(1, height, width, 1), (1 - x).expand(1, height, width, 1)), dim=3)
        return torch.cat([torch.roll(base, shifts=index * 19, dims=2) for index in range(frames)])

    def test_frame_interpolation_node(self):
        result = self.nodes.NvidiaDLSSFrameInterpolation.execute(
            self.nodes.InputImpl.VideoFromFile(str(self.source)),
            "60", "Auto", "Max", "H.264", "MP4", "Auto", "_DLSSFG", False,
        )
        report = self.check_video(result, (640, 360), 24, 60)
        self.assertGreater(report["generated_frames"], 0)

    def test_image_batch_neural_rendering(self):
        torch = self.torch
        image = torch.linspace(0, 1, 360 * 640 * 3).reshape(1, 360, 640, 3)
        image = torch.cat((image, image.flip(2)), dim=0)
        result = self.nodes.NvidiaDLSSImageUpscale.execute(
            image, "1.5× (Quality)", False, "Default", "Default",
            1.0, 1.0, 1.0, -1.0, False, "Default",
            output_detail_strength=2.0,
        )
        self.assertEqual(tuple(result.result[0].shape), (2, 540, 960, 3))
        self.assertTrue(torch.isfinite(result.result[0]).all())
        report = json.loads(result.result[1])
        self.assertTrue(report["feature_18_confirmed"])
        self.assertEqual(report["output_composition"]["output_detail_strength"], 2.0)
        self.assertTrue(report["output_composition"]["applied"])
        self.assertEqual(report["images_processed"], 2)
        self.assertTrue(
            report["nr_upscaling_active"] or report["nr_native_fallback"]
        )

    def test_video_neural_rendering(self):
        result = self.nodes.NvidiaDLSSVideoUpscale.execute(
            self.nodes.InputImpl.VideoFromFile(str(self.source)),
            "1.5× (Quality)", False, "Default", "Default",
            1.0, 1.0, 1.0, -1.0, False, "Default", "Max", "H.264", "MP4",
            "Auto", "_DLSSNR", False,
            output_detail_strength=2.0,
        )
        report = self.check_video(result, (960, 540), 12, 30)
        self.assertTrue(report["feature_18_confirmed"])
        self.assertEqual(report["output_composition"]["output_detail_strength"], 2.0)
        self.assertTrue(report["output_composition"]["applied"])
        self.assertTrue(
            report["nr_upscaling_active"] or report["nr_native_fallback"]
        )

    def test_frame_interpolation_image_batch(self):
        result = self.nodes.NvidiaDLSSImageFrameInterpolation.execute(
            self.image_batch(), "30", "60", "Auto",
        )
        output = result.result[0]
        self.assertEqual(tuple(output.shape), (10, 360, 640, 3))
        self.assertTrue(self.torch.isfinite(output).all())
        report = json.loads(result.result[2])
        self.assertEqual(report["source_type"], "image_sequence")
        self.assertEqual(report["input_frames"], 5)
        self.assertEqual(report["output_frames"], 10)
        self.assertEqual(result.result[1], 60.0)
        self.assertGreater(report["generated_frames"], 0)

    def test_frame_interpolation_default_input_fps_24(self):
        result = self.nodes.NvidiaDLSSImageFrameInterpolation.execute(
            self.image_batch(), "24", "60", "Auto",
        )
        report = json.loads(result.result[2])
        self.assertEqual(report["input_fps"], "24")
        self.assertEqual(result.result[1], 60.0)

    def test_frame_interpolation_exact_fractional_output_fps(self):
        result = self.nodes.NvidiaDLSSImageFrameInterpolation.execute(
            self.image_batch(), "24", "59.94", "Auto",
        )
        report = json.loads(result.result[2])
        self.assertEqual(report["output_fps"], "60000/1001")
        self.assertAlmostEqual(result.result[1], 59.94, delta=0.001)

    def test_image_sequence_neural_rendering(self):
        result = self.nodes.NvidiaDLSSImageSequenceUpscale.execute(
            self.image_batch(frames=3), "1.5× (Quality)", False,
            "Default", "Cinematic", 2.0, 1.0, 1.0, -1.0, False, "Default",
            output_detail_strength=2.0,
        )
        output = result.result[0]
        self.assertEqual(tuple(output.shape), (3, 540, 960, 3))
        self.assertTrue(self.torch.isfinite(output).all())
        report = json.loads(result.result[1])
        self.assertEqual(report["source_type"], "image_sequence")
        self.assertTrue(report["feature_18_confirmed"])
        self.assertEqual(report["output_composition"]["output_detail_strength"], 2.0)
        self.assertTrue(report["output_composition"]["applied"])
        self.assertEqual(report["images_processed"], 3)
        self.assertTrue(report["nr_upscaling_active"] or report["nr_native_fallback"])


if __name__ == "__main__":
    unittest.main()
