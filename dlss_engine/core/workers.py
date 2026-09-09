"""Launch the bundled Windows workers without moving Python into Wine."""

from __future__ import annotations

import atexit
import json
import os
import shutil
import signal
import subprocess
import threading
import time
import sys
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / "linux-runtime.json"
LINUX_ENV_KEYS = frozenset(
    {
        "WINEPREFIX",
        "DLSS_WINE_PATH",
        "WINESERVER",
        "WINEDEBUG",
        "WINE_HEAP_DELAY_FREE",
        "WINE_HEAP_ZERO_MEMORY",
        "WINEDLLOVERRIDES",
        "NVIDIA_WINE_DLL_DIR",
        "DXVK_ENABLE_NVAPI",
        "DXVK_CONFIG",
        "DXVK_FILTER_DEVICE_NAME",
        "VKD3D_FILTER_DEVICE_NAME",
    }
)


def linux_worker_environment() -> dict[str, str]:
    """Apply local setup only to workers, never to ComfyUI's environment."""
    env = os.environ.copy()
    try:
        raw = CONFIG_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return env  # Existing manual environment setup remains supported.
    except OSError as exc:
        raise RuntimeError(
            f"Cannot read Linux settings: {CONFIG_PATH}"
        ) from exc
    try:
        config = json.loads(raw)
        if (
            not isinstance(config, dict)
            or config.get("version") != 1
            or set(config) != {"version", "environment"}
        ):
            raise ValueError("expected version 1 and environment")
        settings = config["environment"]
        if not isinstance(settings, dict) or set(settings) != LINUX_ENV_KEYS:
            setting_keys = set(settings) if isinstance(settings, dict) else set()
            missing = sorted(LINUX_ENV_KEYS - setting_keys)
            unexpected = sorted(setting_keys - LINUX_ENV_KEYS)
            detail = []
            if missing:
                detail.append("missing " + ", ".join(missing))
            if unexpected:
                detail.append("unexpected " + ", ".join(unexpected))
            raise ValueError(
                "environment must contain the complete helper-generated key set"
                + (" (" + "; ".join(detail) + ")" if detail else "")
            )
        for key, value in settings.items():
            if (
                key not in LINUX_ENV_KEYS
                or not isinstance(value, str)
                or "\0" in value
            ):
                raise ValueError(f"invalid setting: {key}")
        # Saved setup is authoritative: unrelated Wine settings inherited
        # from a desktop or shell must not select a different prefix/runtime.
        env.update(settings)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            f"Invalid Linux settings in {CONFIG_PATH}: {exc}. "
            "Rerun scripts/setup_linux.py or remove the file for manual setup."
        ) from exc
    return env


def validate_binary(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"DLSS runtime file is missing: {path}")
    with path.open("rb") as stream:
        header = stream.read(64)
    if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"{path.name} is a Git LFS pointer. Run git lfs pull in the "
            "custom-node directory to download the runtime binaries."
        )
    if not header.startswith(b"MZ"):
        raise RuntimeError(f"Not a Windows runtime binary: {path}")


def worker_launch(path: Path, *args: str) -> tuple[list[str], dict]:
    validate_binary(path)
    command = [str(path.resolve()), *args]
    if sys.platform == "win32":
        return command, {
            "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
        }
    if sys.platform != "linux":
        raise RuntimeError("DLSS workers support Windows or Linux with Wine.")

    env = linux_worker_environment()
    prefix = env.get("WINEPREFIX", "")
    if not prefix or not Path(prefix).is_absolute():
        raise RuntimeError(
            "Linux DLSS requires WINEPREFIX to point to an existing, "
            "configured Wine prefix using an absolute path. See docs/linux.md."
        )
    if not (Path(prefix) / "drive_c/windows/system32").is_dir():
        raise RuntimeError(
            "WINEPREFIX is not initialized. Set up Wine, DXVK, "
            "VKD3D-Proton and DXVK-NVAPI as described in docs/linux.md."
        )
    configured = env.get("DLSS_WINE_PATH")
    wine = (
        shutil.which(configured)
        if configured
        else (shutil.which("wine64") or shutil.which("wine"))
    )
    if not wine:
        raise RuntimeError(
            "Wine was not found. Install Wine or set DLSS_WINE_PATH to "
            "its executable (a path, not a shell command)."
        )
    # Wine diagnostics belong on stderr, never on the binary protocol stream.
    env.setdefault("WINEDEBUG", "-all")
    if not env.get("DISPLAY"):
        # Headless hosts: DXVK needs an X11 connection to create its Vulkan
        # instance (VK_KHR_win32_surface); without one the worker hangs on
        # surface creation. Share one virtual framebuffer across all workers
        # of this process instead of wrapping every launch in xvfb-run, which
        # would add noise to the binary protocol streams.
        env["DISPLAY"] = _ensure_virtual_display()
    return [wine, *command], {"env": env, "start_new_session": True}


_VIRTUAL_DISPLAY_LOCK = threading.Lock()
_virtual_display_server: subprocess.Popen | None = None


def _ensure_virtual_display() -> str:
    global _virtual_display_server
    with _VIRTUAL_DISPLAY_LOCK:
        if (
            _virtual_display_server is not None
            and _virtual_display_server.poll() is None
        ):
            return _virtual_display_server.args[1]
        xvfb = shutil.which("Xvfb")
        if xvfb is None:
            raise RuntimeError(
                "No X display found and the \"xvfb\" package is not installed. "
                "Install it (e.g. apt install xvfb) or start ComfyUI under a "
                "virtual display such as xvfb-run."
            )
        last_error = None
        for number in range(90, 105):
            display = f":{number}"
            try:
                server = subprocess.Popen(
                    [xvfb, display, "-screen", "0", "640x480x24"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except OSError as exc:
                last_error = exc
                continue
            # Wait for the socket to appear or the server to die (e.g. the
            # display number is already taken). A bare poll() right after
            # spawn races against Xvfb startup and misreports success.
            deadline = time.monotonic() + 5.0
            socket_path = Path(f"/tmp/.X11-unix/X{number}")
            while True:
                if server.poll() is not None:
                    break
                if socket_path.exists():
                    _virtual_display_server = server
                    atexit.register(_stop_virtual_display)
                    return display
                if time.monotonic() > deadline:
                    try:
                        os.killpg(server.pid, signal.SIGTERM)
                    except (ProcessLookupError, OSError):
                        pass
                    break
                time.sleep(0.05)
            last_error = RuntimeError(f"Xvfb exited with {server.returncode}")
        raise RuntimeError(
            "No X display found and no virtual framebuffer could be started"
            + (f" ({last_error})" if last_error else "")
            + ". Install the \"xvfb\" package or start ComfyUI under a "
            "virtual display such as xvfb-run."
        )


def _stop_virtual_display() -> None:
    global _virtual_display_server
    server = _virtual_display_server
    if server is not None and server.poll() is None:
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
    _virtual_display_server = None


class WorkerProcess(subprocess.Popen):
    """Own a render's process group, including children of its Wine loader."""

    def __init__(self, path: Path, *args: str, **kwargs) -> None:
        command, options = worker_launch(path, *args)
        self._owns_group = bool(options.get("start_new_session"))
        super().__init__(command, **options, **kwargs)

    def terminate(self) -> None:
        if self._owns_group:
            self._signal_group(signal.SIGTERM)
        else:
            super().terminate()

    def kill(self) -> None:
        if self._owns_group:
            self._signal_group(signal.SIGKILL)
        else:
            super().kill()

    def _signal_group(self, sig: int) -> None:
        # Never use wineserver -k: it also kills unrelated prefix clients.
        try:
            os.killpg(self.pid, sig)
        except ProcessLookupError:
            pass
