# ComfyUI NVIDIA DLSS 5 Visual Enhancer

Native NVIDIA DLSS processing for ComfyUI image and video workflows.

This release keeps the existing `VIDEO` nodes and adds two IMAGE-sequence nodes.

- **NVIDIA DLSS Frame Interpolation** — `VIDEO → VIDEO`
- **NVIDIA DLSS Video Upscale** — `VIDEO → VIDEO`
- **NVIDIA DLSS Image Frame Interpolation** — `IMAGE` batch → higher-rate `IMAGE` batch.
- **NVIDIA DLSS Image Sequence Upscale** — `IMAGE` batch → larger `IMAGE` batch.
- **NVIDIA DLSS Image Upscale** — `IMAGE` batch → larger `IMAGE` batch.

```text
VIDEO input -> NVIDIA DLSS Frame Interpolation -> Save Video
VIDEO input -> NVIDIA DLSS Video Upscale -> Save Video
IMAGE sequence -> NVIDIA DLSS Image Frame Interpolation -> Video Combine / Save Image
IMAGE sequence -> NVIDIA DLSS Image Sequence Upscale -> Video Combine / Save Image
IMAGE sequence -> NVIDIA DLSS Image Upscale -> Save Image
```

The VIDEO nodes use temporary transcoding as needed by the worker. The IMAGE
nodes keep frame order in memory, so a downstream ComfyUI node remains
responsible for encoding, audio, subtitles, timing metadata, and the permanent
output path.

## Workflow compatibility

Existing workflows using **VIDEO** nodes keep working with their current node IDs.
IMAGE-sequence nodes are new and should be used for image-first pipelines
(typically after **VAE Decode**). Image-based interpolation has an explicit **Input
FPS** control because image batches do not carry embedded container timing.



## Requirements

- A current ComfyUI installation.
- A compatible NVIDIA RTX GPU and current NVIDIA display driver.
- Windows, or Linux x86-64 with Wine (experimental).
- Git LFS for source clones, because the bundled runtime binaries use LFS.

The custom node includes the DLSS runtimes, RenoDX/ReShade carrier, and worker
executables. Current ComfyUI supplies Python dependencies including PyTorch,
NumPy, and OpenCV. FFmpeg and FFprobe must be available on `PATH` for the
preserved VIDEO nodes, which probe, decode, encode, and mux their temporary
outputs. IMAGE-only workflows do not call them until a downstream node encodes
the returned batch into video. Set `DLSS_FFMPEG_PATH` and `DLSS_FFPROBE_PATH`
to explicit Linux executable paths when they are not on `PATH`.

## Installation

1. Clone with Git LFS, or install a complete release archive:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/Konohamaru04/ComfyUI-NVIDIA-DLSS-Frame-Interpolation.git
   cd ComfyUI-NVIDIA-DLSS-Frame-Interpolation
   git lfs pull
   ```

2. Restart ComfyUI.

3. On Linux, run the [setup helper](docs/linux.md#setup-helper-recommended)
   once, then launch ComfyUI normally. It saves the Wine worker configuration
   in `linux-runtime.json`; a separate launch script is not needed.

## NVIDIA DLSS Image Frame Interpolation

Connect an ordered batch with at least two frames. Choose the real source rate
with **Input FPS**, then select **Output FPS** and a DLSS engine. Fractional
choices use exact 1001-based rates. The output is a new `IMAGE` batch on the
requested constant-rate timeline.

| Control | Purpose |
| --- | --- |
| Input FPS | Rate of the ordered input images. VAE Decode does not provide this itself. |
| Output FPS | `23.976` through `120`; target must not exceed 6× input rate. |
| DLSS Engine | `Auto` uses native DLSSG where its grid is exact, otherwise it cascades 2× stages. |

`report_json` records input/output counts, copied/generated frames, timing
error, scene-cut resets, chosen plan, and detected GPU/runtime capability.

## NVIDIA DLSS Image Sequence Upscale

This node processes every image in order and returns an upscaled `IMAGE` batch.
It keeps temporal guide history across frames instead of resetting the DLSS
session for every image. A single frame is valid; a batch receives temporal
guides from its preceding images.

- RGB input returns RGB output.
- RGBA input returns RGBA output, with alpha resized using Lanczos.
- Single-channel input returns RGB output.
- Every input image must be at least 64×64 pixels.

Upscale modes range from `1x (DLAA / native)` through `3x (Ultra Performance)`.
The node exposes the DLSS Neural Rendering controls: preset, style, intensity,
tone/structure controls, automatic mask, model preset, and optional SDR output
detail strength.

### Neural-upscaling status

Read the `report_json` fields rather than treating larger dimensions as proof
of neural Super Resolution:

- `feature_18_confirmed` means the signed DLSS Neural Rendering feature ran.
- `nr_upscaling_active` means it accepted the lower-resolution neural-upscale contract.
- `nr_native_fallback` means that contract was rejected and NR ran at output
  resolution after the carrier's DLSS SR pass.

**Require Neural Upscaling** changes this into a strict pass/fail check. It
does not make an unsupported driver/runtime path active.

## Linux

Follow the detailed [Linux guide](docs/linux.md). It documents a dedicated
Wine prefix with GE-Proton, DXVK, VKD3D-Proton, DXVK-NVAPI, the native shader
compiler, NVIDIA's Wine NGX bridge, and the patched Linux ReShade carrier.

The setup was tested on CachyOS with an RTX 5090, NVIDIA driver **610.57.04**,
and GE-Proton 11-6. Native/cascaded IMAGE frame interpolation and image-sequence
NR succeeded. On that driver, direct NR upscaling returns `0xBAD00005`; the
supported fallback runs DLSS SR followed by native-resolution NR and reports
`nr_native_fallback: true`. Leave **Require Neural Upscaling** off to accept
that fallback, or enable it when your workflow must reject it.

## Validation

The test suite covers deterministic IMAGE-timeline selection, cancellation,
temporal image-sequence upscaling, and opt-in GPU tests through the actual
ComfyUI node entry points. The GPU test feeds a synthetic ordered `IMAGE` batch
through 30→60 FPS interpolation and 1.5× Cinematic sequence upscale, then
checks output tensor dimensions, finite values, feature-18 evidence, and
fallback status.

Run all non-GPU tests:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

With configured Linux workers and ComfyUI's Python environment, run the real
Comfy node test:

```bash
DLSS_COMFYUI_PATH="$HOME/ComfyUI" DLSS_RUN_COMFY_GPU_TESTS=1 \
  "$HOME/ComfyUI/.venv/bin/python" -m unittest discover \
  -s tests -p 'test_comfy_gpu.py'
```

## Credits and licenses

The implementation is derived from [DLSS 5 Visual Enhancer](https://github.com/Merserk/dlss5-visual-enhancer)
v5.0 by Merserk. Its MIT license is included as
[`LICENSE-DLSS-Visual-Enhancer.txt`](LICENSE-DLSS-Visual-Enhancer.txt).

Runtime licenses are included beside their files under `bin/runtime/host`,
`bin/runtime/dlss`, and `bin/runtime/dlssg`. Review the NVIDIA DLSS license
before redistribution or commercial release. RenoDX and ReShade license texts
are included under `bin/runtime/host`.
