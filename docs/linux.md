# Linux setup (experimental)

The Python nodes and PyTorch run natively on Linux. The existing Windows DLSS
workers run in a dedicated Wine prefix and exchange ordered IMAGE frames with
Python over binary pipes. No new Python runtime dependencies are introduced.

## Tested behavior

Tested on CachyOS x86-64, an NVIDIA GeForce RTX 5090, driver **610.57.04**,
and the Wine/DXVK/VKD3D-Proton/DXVK-NVAPI components from **GE-Proton 11-6**.
The Linux setup requires the patched ReShade carrier documented below.

- Frame interpolation: native and cascaded 640×360 IMAGE batches, 30→60 FPS,
  with 12 input images and 24 returned images.
- Image-sequence upscale: two 640×360 images produced two 960×540 images with
  confirmed DLSSNR feature-18 execution.

The tested upscale paths execute DLSS SR followed by NR at output resolution.
The runtime reports `nr_native_fallback: true` and `nr_upscaling_active: false`.
This is real NR processing, but it is **not direct NR reconstruction from the
lower-resolution input**. The existing **Require Neural Upscaling** option
continues to reject that fallback. See [NR behavior](#nr-behavior).

Other Wine builds, GPUs, drivers, HDR settings, and resolutions need
independent validation. Plain Wine 11.16 with Proton Experimental graphics
libraries also passed interpolation; the complete recipe below uses GE-Proton.
No NVIDIA runtime or worker executable is modified.

## Setup helper (recommended)

The optional [setup helper](../scripts/setup_linux.py) automates sections 3–6
below. First install the prerequisites, clone this custom node with Git LFS,
and install GE-Proton. It uses Python's standard library; it does not install
system packages, download DLLs, or require root.

You need Microsoft's **x86-64 `d3dcompiler_47.dll`**. The helper looks in
`~/.cache/winetricks/d3dcompiler_47/d3dcompiler_47.dll`, which may already exist
from a previous Winetricks installation. Otherwise obtain the native compiler
using the [Winetricks recipe below](#native-shader-compiler-and-heap-settings)
in a separate prefix, then supply its cached DLL with `--compiler /path/to/d3dcompiler_47.dll`.
Do not pass Wine's builtin compiler. The helper validates PE architecture;
that check alone does not establish compiler origin or working NR.

From this custom-node directory, adjusting paths and GPU name:

```bash
python3 scripts/setup_linux.py \
  --proton "$HOME/.local/share/Steam/compatibilitytools.d/GE-Proton11-6-x86_64" \
  --comfyui "$HOME/ComfyUI" \
  --prefix "$HOME/.local/share/comfyui-dlss-helper" \
  --gpu 'NVIDIA GeForce RTX 5090' \
  --verify
```

Use `--check` instead of `--verify` to validate inputs and preview the paths
without writing anything. Omit both to configure without rendering tests.
With exactly one detected GPU, `--gpu` is optional. `--python` selects ComfyUI's
Python if it is not at `COMFYUI/.venv/bin/python`. `--driver-dir` selects the
installed NVIDIA Wine bridge directory on other distributions. Run
`python3 scripts/setup_linux.py --help` for every option.

The helper creates a dedicated prefix, installs the matching GE-Proton
DXVK/VKD3D-Proton/NVAPI DLLs and native compiler, links the installed driver
bridge, and saves the heap flags, DLL overrides and GPU selection in the
custom node’s local `linux-runtime.json`. Start ComfyUI normally:

```bash
cd "$HOME/ComfyUI"
source .venv/bin/activate
python main.py
```

Rerun the same setup command to refresh DLLs or saved settings. It only
accepts a new/empty prefix or one previously marked by this helper. It refuses
to modify an existing manual or game prefix; select a new directory instead.
It also stops on conflicting node-local driver links so you can inspect them.
An interrupted first initialization can be retried, and reruns do not execute
`wineboot` on an already initialized prefix. Stop renders before updating the
prefix. The settings file is ignored by Git and contains machine-specific paths.

Only Linux worker subprocesses receive these settings. ComfyUI, PyTorch,
FFmpeg and other custom nodes retain their normal environment. Saved settings
take precedence over inherited Wine variables, preventing an unrelated shell
or desktop Wine prefix from overriding the configured runtime. Without the
file, the manual environment setup below remains supported. Windows does not
read this file. Invalid JSON, unsupported versions and unknown environment
keys produce an actionable error instead of silently falling back.

If you used the earlier helper that generated `start-comfyui.sh`, rerun setup
with the same prefix to save the settings, then restart ComfyUI using your
normal command or desktop shortcut. The old launcher is no longer needed;
setup leaves existing scripts untouched. Moving this custom-node directory to
another machine requires running setup there with its own paths.

`--verify` runs both real ComfyUI IMAGE-node tests, including feature-18 log
verification, output dimensions, frame counts, and timing checks.
Failure returns a nonzero exit status; it does not claim that configuration
alone proves NR works. These tests retain the direct-NR/fallback distinction
and do not establish subjective enhancement quality.

The helper was tested from a fresh prefix on the RTX 5090 / driver 610.57.04
with GE-Proton 11-6: both ComfyUI IMAGE-node tests passed. Automated tests also
cover reruns, destination symlinks, saved paths containing spaces/quotes,
worker-only environment handling, invalid settings, check-only behavior,
unmanaged-prefix refusal and invalid/LFS-pointer DLLs.

## 1. Prerequisites

Use a working native Linux ComfyUI installation and its Python environment.
Install a recent Wine build, Git LFS, FFmpeg/FFprobe, and Vulkan tools through
your distribution. For example, on Arch/CachyOS:

```bash
sudo pacman -S --needed wine git-lfs ffmpeg vulkan-tools
```

The NVIDIA driver must include its Vulkan and NGX userspace components, not
only CUDA compute support. Confirm both tools see your intended GPU:

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
vulkaninfo --summary
ffmpeg -version
ffprobe -version
```

Use a desktop session for the initial setup. Headless operation has not been
validated. See [WineHQ installation](https://gitlab.winehq.org/wine/wine/-/wikis/Download)
for distributions whose packaged Wine is older.

## 2. Install the complete custom node

Replace the ComfyUI path if needed:

```bash
cd "$HOME/ComfyUI/custom_nodes"
git clone https://github.com/Konohamaru04/ComfyUI-NVIDIA-DLSS-Frame-Interpolation.git
cd ComfyUI-NVIDIA-DLSS-Frame-Interpolation
git lfs install --local
git lfs pull
```

For a PR checkout, use its branch before the final `git lfs pull`. A GitHub
source ZIP can contain LFS pointers instead of the runtime binaries. The
launcher reports this explicitly rather than trying to execute pointer text.

## 3. Initialize a dedicated Wine prefix

Install [GE-Proton 11-6](https://github.com/GloriousEggroll/proton-ge-custom/releases/tag/GE-Proton11-6)
and adjust its directory below to match your installation. This recipe uses
its Wine executable directly, not the Steam or UMU launcher. Run these
commands in the terminal you will later use to start ComfyUI:

```bash
export WINEPREFIX="$HOME/.local/share/comfyui-dlss-wine"
export DLSS_PROTON_FILES="$HOME/.local/share/Steam/compatibilitytools.d/GE-Proton11-6-x86_64/files"
export DLSS_WINE_PATH="$DLSS_PROTON_FILES/bin/wine"
export WINESERVER="$DLSS_PROTON_FILES/bin/wineserver"
mkdir -p "$WINEPREFIX"
"$DLSS_WINE_PATH" wineboot -u
"$DLSS_WINE_PATH" cmd /c echo Wine-ready
```

Wait for Wine initialization to finish before installing the graphics DLLs.
Running `wineboot -u` again can replace installed DLLs; reinstall the following
components if that happens. `WINEPREFIX` must be an absolute path pointing to
an initialized prefix. The nodes do not create or configure a prefix automatically; use the helper
or this manual recipe.

`DLSS_WINE_PATH` is a single executable path, including paths containing spaces.
It is not a shell command, a string of arguments, or Proton's Python launcher.
If omitted, the node looks for `wine64`, then `wine`, on `PATH`.

## 4. Install the graphics translation libraries

You need x86-64 builds of [DXVK](https://github.com/doitsujin/dxvk),
[VKD3D-Proton](https://github.com/HansKristian-Work/vkd3d-proton), and
[DXVK-NVAPI](https://github.com/jp7677/dxvk-nvapi). Follow their installation
instructions or use a coherent set bundled with Proton. The following
recipe uses the same GE-Proton installation selected above:

```bash
export DLSS_WINE_LIBS="$DLSS_PROTON_FILES/lib/wine"
export DLSS_SYSTEM32="$WINEPREFIX/drive_c/windows/system32"

cp --remove-destination "$DLSS_WINE_LIBS/dxvk/x86_64-windows/"{dxgi,d3d11}.dll "$DLSS_SYSTEM32/"
cp --remove-destination "$DLSS_WINE_LIBS/vkd3d-proton/x86_64-windows/"{d3d12,d3d12core}.dll "$DLSS_SYSTEM32/"
cp --remove-destination "$DLSS_WINE_LIBS/nvapi/x86_64-windows/"{nvapi64,nvofapi64}.dll "$DLSS_SYSTEM32/"

export WINEDLLOVERRIDES='dxgi,d3d11,d3d12,d3d12core,nvapi64,nvofapi64=n'
export DXVK_CONFIG='dxgi.hideNvidiaGpu = False'
export DXVK_ENABLE_NVAPI=1
```

Keep these DLLs and the selected Wine build from the same GE-Proton release. These are Bash
commands; the braces expand the individual DLL names. `--remove-destination`
replaces prefix symlinks instead of writing through them into system files.

The DXVK setting matters: if NVIDIA is hidden behind an AMD compatibility
identity, the worker reports **No NVIDIA D3D12 adapter was found** even though
the log contains the correct GPU name. If you have other DXVK settings or DLL
overrides, combine them with the settings above.

### Native shader compiler and heap settings

NR requires Microsoft's native shader compiler. Install Winetricks through
your distribution, then install the compiler in this prefix:

```bash
WINE="$DLSS_WINE_PATH" WINEPREFIX="$WINEPREFIX" winetricks -q d3dcompiler_47
export WINEDLLOVERRIDES='dxgi,d3d11,d3d12,d3d12core,nvapi64,nvofapi64,d3dcompiler_47=n,b'
export WINE_HEAP_DELAY_FREE=1
export WINE_HEAP_ZERO_MEMORY=1
```

The heap flags are the Wine settings applied by Proton's
`PROTON_HEAP_DELAY_FREE` and `PROTON_HEAP_ZERO_MEMORY` options. Export the
`WINE_*` names because this integration invokes Wine directly. Without them,
the tested NR worker could render a frame and then crash during cleanup.

On the tested 64-bit-only prefix, Winetricks copied the x86-64 compiler DLL
but its final 32-bit `regedit.exe` invocation failed. Verify that
`$WINEPREFIX/drive_c/windows/system32/d3dcompiler_47.dll` exists and is x86-64;
the explicit DLL override selects it without that registry step. Do not treat
other installation failures as successful installation.

### ReShade descriptor compatibility fix

Linux selects `bin/runtime/host-linux/dxgi.dll`, the build containing the
[VKD3D descriptor fix](../patches/README.md). The original ReShade extension
hooks also translate descriptors allocated on an unwrapped device, causing
NR to fault inside a CUDA descriptor lookup. The patch preserves native
handles while continuing to translate ReShade's virtual handles.

The source revision, patch, license, and GitHub Actions build are included
so the modified carrier can be reviewed and rebuilt. This changes ReShade,
not any NVIDIA DLL. Do not replace it with an ordinary ReShade download or
with the DXVK DLL copied into `system32`.

The bundled carrier is ReShade `6.8.0.1` (unofficial), produced by
[this source build](https://github.com/sujaisubbanna/ComfyUI-NVIDIA-DLSS-Frame-Interpolation/actions/runs/33919628924).
Its SHA-256 is
`596e4a61b96540683fdb92f80c72c96637247276a34615c3c26a35ba78725e80`.
Verify it with `sha256sum bin/runtime/host-linux/dxgi.dll` from the node directory.

Windows continues to use the unchanged upstream `bin/runtime/host/dxgi.dll`
(SHA-256 `0cee63f9c9f13f3ac909c5b4903f4dbb4b719a7ab3b4f13b0deaf83c814b94f7`).
Before the first Linux render, Python copies the small worker executable into
`host-linux/` and links the shared add-on, NR DLL and local driver bridge there.
This makes Wine load the Linux carrier beside the worker. The Linux host's
`ReShade.ini` and `ReShade.log` are separate, locally generated files; the
worker writes its NR settings on each run. The node directory must be writable.
No NVIDIA binary is duplicated or downloaded during preparation.

## 5. Make NVIDIA's NGX bridge discoverable

Locate `_nvngx.dll` from your installed NVIDIA Linux driver. On the tested
Arch/CachyOS system it is `/usr/lib/nvidia/wine/_nvngx.dll`. Other distributions
may install it elsewhere. The matching Linux `libnvidia-ngx.so` must also be
installed and discoverable by the system loader.

From the **custom-node repository directory**:

```bash
export NVIDIA_WINE_DLL_DIR=/usr/lib/nvidia/wine
test -f "$NVIDIA_WINE_DLL_DIR/_nvngx.dll"
ln -s "$NVIDIA_WINE_DLL_DIR/_nvngx.dll" bin/runtime/dlssg/_nvngx.dll
ln -s "$NVIDIA_WINE_DLL_DIR/_nvngx.dll" bin/runtime/host/_nvngx.dll
```

These links are local setup files excluded from Git. If they already exist,
check their targets instead of rerunning `ln`. Keep the bridge matched to your
installed driver; do not download an unrelated Windows driver DLL.

The workers search their own directories for `_nvngx.dll`. With the tested
plain Wine build, placing it only in the prefix's `system32`, or setting only
`NVIDIA_WINE_DLL_DIR`, did not satisfy the worker's NGX loader. The local links
resolved `NVSDK_NGX_D3D12_Init_with_ProjectID failed: 0xBAD00001`.

**Do not replace `bin/runtime/host/nvngx.dll`.** Despite its name, that bundled
file is the enhancement worker executable. Likewise, retain the bundled
`bin/runtime/host-linux/dxgi.dll`: it is the ReShade carrier, not the DXVK DLL installed
in the prefix. Driver bridges are not redistributed by this project.

## 6. Start native ComfyUI

If you used the helper, launch ComfyUI normally; the settings are already saved.
The environment instructions in this section apply to manual setup only.

On a multi-GPU machine, select the same NVIDIA adapter for DXVK and
VKD3D-Proton. For example:

```bash
export DXVK_FILTER_DEVICE_NAME='NVIDIA GeForce RTX 5090'
export VKD3D_FILTER_DEVICE_NAME='NVIDIA GeForce RTX 5090'
```

These filters select the Windows worker's Vulkan adapter; `CUDA_VISIBLE_DEVICES`
alone does not do that. The existing node uses automatic GPU selection, so
check worker logs on multi-GPU systems rather than relying only on the Python
GPU label in the report.

For manual setup, start ComfyUI from the same terminal so workers inherit the
prefix and overrides:

```bash
cd "$HOME/ComfyUI"
source .venv/bin/activate
python main.py
```

Use your actual environment/launch command if it differs. Keep using native
Linux FFmpeg and FFprobe. The existing `DLSS_FFMPEG_PATH` and
`DLSS_FFPROBE_PATH` overrides accept Linux executable paths too.

## Verify frame interpolation

Before starting ComfyUI, with its Python environment activated and either
saved settings or the manual Wine variables, run this from the custom-node
directory:

```bash
python - <<'PY'
from dataclasses import asdict
import json
from dlss_engine.frame_interpolation.capabilities import probe_frame_interpolation_capabilities
print(json.dumps(asdict(probe_frame_interpolation_capabilities()), indent=2))
PY
```

Expect `available: true` and `native_multiplier` greater than one. HAGS and
Windows Authenticode checks are not applicable on Linux; they are not bypasses
for the runtime's actual capability checks.

Then route an ordered IMAGE batch from VAE Decode into **NVIDIA DLSS Image
Frame Interpolation**, set Input FPS to 30, Output FPS to 60, and Engine to
Auto. Inspect the returned IMAGE batch and `report_json`; use a downstream
video-combine or image-save node when output files are needed.

## NR behavior

Image-sequence enhancement uses RenoDX/ReShade and the separate signed DLSSNR
feature-18 runtime. With the setup above, the sequence node completed actual
NR evaluation. Its report distinguishes these cases:

- `feature_18_confirmed: true`: signed NR execution was verified in the logs.
- `nr_upscaling_active: true`: NR directly reconstructed the larger output.
- `nr_native_fallback: true`: the runtime rejected direct NR upscaling and
  evaluated NR at output resolution after the carrier's DLSS SR pass.

On the tested RTX 5090 / NVIDIA Linux driver **610.57.04** configuration,
direct NR upscaling returned `0xBAD00005` and the existing native-resolution
NR fallback succeeded. The output was larger and NR executed, but
`nr_upscaling_active` remained false. This driver/runtime combination therefore
does **not** provide direct neural upscaling on Linux. Leave **Require Neural
Upscaling** at its default `false` to accept the existing fallback policy.
Setting it to `true` intentionally rejects such output. Neither the Linux
launcher nor the compatibility patch changes that check.

### Proton environment variables

[Proton's NVAPI support](https://github.com/GloriousEggroll/proton-ge-custom)
uses `PROTON_ENABLE_NVAPI=1` and `PROTON_HIDE_NVIDIA_GPU=0`. The equivalent
Wine setup above uses the native NVAPI DLLs, `DXVK_ENABLE_NVAPI=1`, and
`dxgi.hideNvidiaGpu = False`. A common ReShade override is `dxgi=n,b`; merge it
with the complete override list instead of discarding the other entries.

`PROTON_*` variables are interpreted by Proton's launcher. Setting them on a
plain Wine command does not apply Proton's setup. Point `DLSS_WINE_PATH` at
a Wine executable, not `proton` or `umu-run`: the worker depends on a clean
binary stdin/stdout protocol.

NVIDIA documents `PROTON_ENABLE_NGX_UPDATER=1` and DLSS override variables in
its [Linux gaming guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/gaming.html).
Those settings did not resolve this carrier's descriptor bug. Forcing NVAPI
or disabling its newer descriptor entry points also did not provide working
NR with the original carrier. The tested fix requires the patched ReShade,
native shader compiler, and Wine heap settings together.

## Troubleshooting and testing

- **NR access violation:** verify the patched carrier is present, the native
  compiler override is active, and both `WINE_HEAP_*` flags are inherited by
  ComfyUI. Keep `bin/runtime/host-linux/ReShade.log` and worker diagnostics.
- **Shader error mentioning `isnan`:** Wine's built-in compiler is being used;
  install and select native `d3dcompiler_47` as above.
- **Wine not found / invalid prefix:** export `DLSS_WINE_PATH` and `WINEPREFIX`
  before launching ComfyUI. A desktop shortcut will not inherit another
  terminal's environment.
- **Probe timeout:** finish Wine initialization first, then verify the graphics
  libraries and driver bridge. The probe has a 45-second timeout and stops its
  own worker process group. It never runs `wineserver -k` against your prefix.
- **Missing DLLs such as MSVCP140/VCRUNTIME140:** install the Visual C++ runtime
  with `winetricks -q vcrun2022` in this prefix, checking installation success.
- **Driver update:** verify the `_nvngx.dll` links still resolve to the new
  driver's bridge and that its Linux NGX library is installed.
- **Need more logs:** set `WINEDEBUG` to `+seh,+loaddll` in the saved environment
  object (or export it for manual setup). It can
  produce large stderr logs; keep it unset for ordinary use.

Regression tests run without a GPU, Wine installation, or downloaded DLSS
binaries. Use the ComfyUI environment, or install `numpy`, `av`, and
`opencv-python-headless` into a separate test environment:

```bash
python -m unittest discover -s tests -v
```

To run the real interpolation test as well, with the setup above active:

```bash
DLSS_RUN_GPU_TESTS=1 python -m unittest discover -s tests -v
```

The GPU test creates an ordered synthetic IMAGE batch, runs interpolation,
checks returned dimensions/frame count/FPS and exercises cancellation. It is
opt-in and is not run by the CPU-only GitHub Actions jobs. It does not validate
subjective interpolation quality.

Python environment and the full NR setup above:
To exercise both actual ComfyUI IMAGE-node entry points, run with ComfyUI's
Python environment and the full NR setup above:

```bash
DLSS_COMFYUI_PATH="$HOME/ComfyUI" DLSS_RUN_GPU_TESTS=1 \
  DLSS_RUN_COMFY_GPU_TESTS=1 python -m unittest discover -s tests -v
```

These tests check returned IMAGE-batch dimensions and finite pixels, frame
counts/FPS, confirmed NR execution, and the distinction between direct NR
upscaling and the runtime's NR fallback.

## SDR composition in ComfyUI

Restart ComfyUI after updating the node. In **NVIDIA DLSS Image Sequence
Upscale**, set `nr_style` to `Cinematic`, `nr_intensity`
to `2`, and the new `output_detail_strength` to `2`. Use `1` for the
unmodified worker output. Old workflows and callers default to `1`.

For an ordered image workflow: VAE Decode → Image Frame Interpolation (60 FPS)
→ Image Sequence Upscale (2× for 1080p→4K, composition 2) → your preferred
video-combine or save node. Check the report for both feature-18 evidence and
`output_composition.applied: true`. The latter reports a CPU output adjustment,
not another GPU inference pass. A successful NR API call alone does not
establish Windows-equivalent quality.

The tested two-second SDR comparison used a direct NR renderer and the actual
OptiScaler composition shader on RTX 5090. The ComfyUI control applies a
bounded brightness-ratio adjustment to the existing worker's output; it does
not install OptiScaler or reproduce its game hooks, exposure measurement,
HDR bridge, or all color controls. Avoid double-applying it to media already
processed at increased output strength.

The bundled model matches the community SF-v2 runtime hash
`6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927`.
The alternate model tested by NapXDD has hash
`e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e`.
A controlled direct-renderer comparison produced identical decoded pixels in
all 60 frames with these two builds. This rules out that DLL difference for
that clip and setup, not all driver/runtime issues. A log mentioning the
signed snippet is not independent verification of the DLL's signature.
