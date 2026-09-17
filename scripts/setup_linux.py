#!/usr/bin/env python3
"""Configure the documented GE-Proton recipe without root or downloads."""

import argparse
import json
import os
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = ".comfyui-dlss-setup"
MARKER_TEXT = "ComfyUI DLSS dedicated prefix v1\n"
GRAPHICS = {
    "dxgi.dll": "wine/dxvk/x86_64-windows/dxgi.dll",
    "d3d11.dll": "wine/dxvk/x86_64-windows/d3d11.dll",
    "d3d12.dll": "wine/vkd3d-proton/x86_64-windows/d3d12.dll",
    "d3d12core.dll": "wine/vkd3d-proton/x86_64-windows/d3d12core.dll",
    "nvapi64.dll": "wine/nvapi/x86_64-windows/nvapi64.dll",
    "nvofapi64.dll": "wine/nvapi/x86_64-windows/nvofapi64.dll",
}


def absolute(value):
    return Path(value).expanduser().absolute()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_pe(path):
    require(path.is_file(), f"Missing DLL: {path}")
    with path.open("rb") as stream:
        header = stream.read(64)
        require(
            not header.startswith(b"version https://git-lfs"),
            f"Git LFS pointer: {path}. Run git lfs pull in {ROOT}.",
        )
        require(
            header[:2] == b"MZ" and len(header) == 64,
            f"Not a PE binary: {path}",
        )
        stream.seek(struct.unpack_from("<I", header, 60)[0])
        require(
            stream.read(6) == b"PE\0\0\x64\x86",
            f"Expected an x86-64 PE binary: {path}",
        )


def run(command, env, timeout=120, cwd=None):
    # Setup owns only its command's group, never the entire Wine server.
    with subprocess.Popen(
        command, env=env, cwd=cwd, start_new_session=True
    ) as process:
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
    require(code == 0, f"Command failed ({code}): {shlex.join(command)}")


def replace_file(source, target):
    # Atomic replacement does not follow Wine's destination DLL symlinks.
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as f:
        temporary = Path(f.name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proton",
        required=True,
        type=absolute,
        help="Installed GE-Proton directory (or its files/)",
    )
    parser.add_argument("--comfyui", type=absolute, default=ROOT.parent.parent)
    parser.add_argument(
        "--python",
        type=absolute,
        help="ComfyUI Python; default: COMFYUI/.venv/bin/python",
    )
    parser.add_argument(
        "--prefix",
        type=absolute,
        default=absolute("~/.local/share/comfyui-dlss-wine"),
    )
    parser.add_argument(
        "--compiler",
        type=absolute,
        default=absolute(
            "~/.cache/winetricks/d3dcompiler_47/d3dcompiler_47.dll"
        ),
        help="Microsoft x64 d3dcompiler_47.dll, defaults to Winetricks cache",
    )
    parser.add_argument(
        "--driver-dir", type=absolute, default=Path("/usr/lib/nvidia/wine")
    )
    parser.add_argument("--gpu", help="Exact NVIDIA GPU name for Vulkan")
    parser.add_argument(
        "--vulkan-name",
        help="Vulkan physical device name for the DXVK/VKD3D-Proton filters. "
        "Defaults to the GPU name. On PRIME-offload systems where the display "
        "runs on another GPU, the NVIDIA driver can report a different Vulkan "
        "deviceName than nvidia-smi (for example 'modesetting'); pass what "
        "vulkaninfo reports, or every adapter gets filtered out.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check inputs and print paths without writing",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After setup, run all three ComfyUI GPU tests",
    )
    args = parser.parse_args(argv)
    require(sys.platform == "linux", "This helper supports Linux only.")
    require(
        not (args.check and args.verify), "Use --check or --verify, not both."
    )
    files = args.proton
    if (files / "files").is_dir():
        files /= "files"
    python = args.python or args.comfyui / ".venv/bin/python"
    wine = files / "bin/wine"
    server = files / "bin/wineserver"
    for executable in (wine, server, python):
        require(
            executable.is_file() and os.access(executable, os.X_OK),
            f"Missing executable: {executable}",
        )
    require(
        (args.comfyui / "main.py").is_file(),
        f"ComfyUI main.py is missing in {args.comfyui}",
    )
    for tool in ("ffmpeg", "ffprobe", "nvidia-smi"):
        require(shutil.which(tool), f"Install {tool} before setup.")
    gpu_names = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=15,
        )
        .strip()
        .splitlines()
    )
    gpu = args.gpu or (gpu_names[0] if len(gpu_names) == 1 else None)
    require(
        gpu in gpu_names,
        "Select a GPU with --gpu. Detected: " + ", ".join(gpu_names),
    )
    # On hybrid systems the Vulkan deviceName can differ from the nvidia-smi
    # name; the DXVK/VKD3D filters must match what the driver reports.
    vulkan_name = args.vulkan_name or gpu

    sources = {name: files / "lib" / path for name, path in GRAPHICS.items()}
    sources["d3dcompiler_47.dll"] = args.compiler
    sources.update(
        {
            p.name: p
            for p in (files / "lib/vkd3d/x86_64-windows").glob("libvkd3d*.dll")
        }
    )
    bridge = args.driver_dir / "_nvngx.dll"
    binaries = [*sources.values(), bridge]
    binaries.extend(
        ROOT / "bin/runtime" / path
        for path in (
            "host/nvngx.dll",
            "host/nvngx_dlssnr.dll",
            "host/renodx-dlss5.addon64",
            "host-linux/dxgi.dll",
            "dlss/nvngx_dlss.dll",
            "dlssg/dlssg-worker.exe",
            "dlssg/nvngx_dlssg.dll",
        )
    )
    for path in binaries:
        check_pe(path)

    marker = args.prefix / MARKER
    owned = marker.is_file() and marker.read_text() == MARKER_TEXT
    require(
        owned
        or not args.prefix.exists()
        or (args.prefix.is_dir() and not any(args.prefix.iterdir())),
        "Refusing to modify an existing unmanaged Wine prefix. "
        "Choose a new --prefix directory; keep game prefixes separate.",
    )
    links = [
        ROOT / "bin/runtime" / host / "_nvngx.dll"
        for host in ("host", "dlssg")
    ]
    for link in links:
        require(
            not link.exists()
            and not link.is_symlink()
            or (link.is_symlink() and link.readlink() == bridge),
            f"Existing bridge differs: {link}. Inspect it before setup.",
        )
    settings = {
        "WINEPREFIX": str(args.prefix),
        "DLSS_WINE_PATH": str(wine),
        "WINESERVER": str(server),
        "WINEDEBUG": "-all",
        "WINE_HEAP_DELAY_FREE": "1",
        "WINE_HEAP_ZERO_MEMORY": "1",
        "WINEDLLOVERRIDES": ",".join(Path(n).stem for n in GRAPHICS)
        + ",d3dcompiler_47=n,b",
        "NVIDIA_WINE_DLL_DIR": str(args.driver_dir),
        "DXVK_ENABLE_NVAPI": "1",
        "DXVK_CONFIG": "dxgi.hideNvidiaGpu = False",
        "DXVK_FILTER_DEVICE_NAME": vulkan_name,
        "VKD3D_FILTER_DEVICE_NAME": vulkan_name,
    }
    config_path = ROOT / "linux-runtime.json"
    print(f"Wine: {wine}\nPrefix: {args.prefix}\nGPU: {gpu}")
    print(f"Vulkan filter: {vulkan_name}")
    print(f"ComfyUI: {args.comfyui}\nSettings: {config_path}", flush=True)
    if args.check:
        print("Input checks passed. No files changed; GPU rendering untested.")
        return
    args.prefix.mkdir(parents=True, exist_ok=True)
    marker.write_text(MARKER_TEXT)
    env = dict(os.environ, **settings)
    system32 = args.prefix / "drive_c/windows/system32"
    if not (args.prefix / "system.reg").is_file():
        # Retrying an interrupted first setup is allowed. Never wineboot an
        # initialized prefix: it can restore builtin graphics DLLs.
        run([str(wine), "wineboot", "-u"], env)
        run([str(wine), "cmd", "/c", "echo", "Wine-ready"], env)
    require(
        system32.is_dir() and not system32.is_symlink(),
        "Wine did not create a regular system32 directory.",
    )
    for name, source in sources.items():
        replace_file(source, system32 / name)
    for link in links:
        if not link.is_symlink():
            link.symlink_to(bridge)
    # Replace the complete settings atomically so a running node can never
    # observe a partial JSON write. This file is local and ignored by Git.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=ROOT, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(
                {"version": 1, "environment": settings}, stream, indent=2
            )
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, config_path)
    finally:
        temporary.unlink(missing_ok=True)
    print("Configured. Start ComfyUI normally; no special launcher is needed.")
    if args.verify:
        # Deliberately remove setup variables: the tests must discover the
        # saved settings just as normally launched ComfyUI does.
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in settings
        }
        env.update(
            DLSS_COMFYUI_PATH=str(args.comfyui), DLSS_RUN_COMFY_GPU_TESTS="1"
        )
        run(
            [
                str(python),
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "test_comfy_gpu.py",
                "-v",
            ],
            env,
            timeout=300,
            cwd=ROOT,
        )
        print("All three ComfyUI GPU tests passed.")
    else:
        print("Rendering not verified. Rerun with --verify to test all nodes.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Setup failed: {exc}\nSee docs/linux.md.", file=sys.stderr)
        sys.exit(1)
