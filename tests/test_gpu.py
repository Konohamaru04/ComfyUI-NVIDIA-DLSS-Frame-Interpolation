"""Opt-in tests against the real worker, driver, encoder and muxer."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import av

from dlss_engine.core.jobs import Cancelled, cancel_active_job
from dlss_engine.core.paths import FFMPEG
from dlss_engine.frame_interpolation import (
    FrameInterpolationOptions,
    interpolate_video,
    interpolate_image_sequence,
)
from fractions import Fraction
import numpy as np


def image_sequence(frames=5, height=360, width=640):
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, None, :, None]
    y = np.linspace(0, 255, height, dtype=np.uint8)[None, :, None, None]
    base = np.empty((1, height, width, 4), dtype=np.uint8)
    base[..., 0:1] = x
    base[..., 1:2] = y
    base[..., 2:3] = 255 - x
    base[..., 3] = 255
    return np.concatenate([np.roll(base, index * 19, axis=2) for index in range(frames)])


@unittest.skipUnless(
    os.environ.get("DLSS_RUN_GPU_TESTS") == "1",
    "Set DLSS_RUN_GPU_TESTS=1 with an RTX GPU and configured runtime",
)
class GPUInterpolationTests(unittest.TestCase):
    def test_interpolate_encode_and_preserve_audio(self):
        with tempfile.TemporaryDirectory(prefix="dlss-gpu-") as directory:
            root = Path(directory)
            source = root / "input.mp4"
            subprocess.run([
                str(FFMPEG), "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i",
                "testsrc2=size=640x360:rate=30:duration=0.4",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=0.4",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest", str(source),
            ], check=True, timeout=30)
            for engine in ("Native DLSSG", "Cascade"):
                with self.subTest(engine=engine):
                    result = interpolate_video(
                        source,
                        FrameInterpolationOptions(
                            target_fps="60", engine=engine,
                        ),
                        output_directory=root / "output",
                        jobs_directory=root / "jobs",
                        logs_directory=root / "logs",
                    )
                    self.assertGreater(result.generated_frames, 0)
                    self.assertEqual(result.output_frames, 24)
                    report = json.loads(Path(result.report_path).read_text())
                    self.assertTrue(report["capabilities"]["available"])
                    with av.open(result.output_path) as video:
                        self.assertEqual(len(video.streams.audio), 1)
                        stream = video.streams.video[0]
                        self.assertEqual(stream.average_rate, 60)
                        frames = list(video.decode(stream))
                        self.assertEqual(len(frames), 24)
                        self.assertEqual(
                            (frames[0].width, frames[0].height), (640, 360)
                        )
            cancelled_output = root / "cancelled-output"

            def cancel_after_first_frame(value, _message):
                if value > 0.05:
                    cancel_active_job()

            with self.assertRaises(Cancelled):
                interpolate_video(
                    source, FrameInterpolationOptions(target_fps="60"),
                    progress=cancel_after_first_frame,
                    output_directory=cancelled_output,
                    jobs_directory=root / "cancelled-jobs",
                    logs_directory=root / "cancelled-logs",
                )
            self.assertEqual(list(cancelled_output.glob("*.mp4")), [])

    def test_interpolate_image_sequence(self):
        result = interpolate_image_sequence(
            image_sequence(), Fraction(30),
            FrameInterpolationOptions(target_fps="60", engine="Auto"),
        )
        self.assertEqual(result.frames.shape, (10, 360, 640, 4))
        self.assertGreater(result.report["generated_frames"], 0)
        self.assertEqual(result.report["source_type"], "image_sequence")
        self.assertTrue(result.report["capabilities"]["available"])

    def test_image_sequence_can_be_cancelled(self):
        def cancel_after_first_frame(value, _message):
            if value > 0.05:
                cancel_active_job()

        with self.assertRaises(Cancelled):
            interpolate_image_sequence(
                image_sequence(), Fraction(30),
                FrameInterpolationOptions(target_fps="60", engine="Auto"),
                progress=cancel_after_first_frame,
            )


if __name__ == "__main__":
    unittest.main()
