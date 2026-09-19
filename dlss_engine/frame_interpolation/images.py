"""In-memory DLSSG interpolation for ordered ComfyUI IMAGE batches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import time
from typing import Callable
from contextlib import suppress

import numpy as np

from ..core.gpu_selection import resolve_runtime_ai_gpu
from ..core.jobs import Cancelled, active_job
from ..core.runtime import prepare_runtime
from .capabilities import probe_frame_interpolation_capabilities
from .models import FrameInterpolationOptions
from .native import DirectDLSSGSession
from .processor import DLSSGStage, TimedFrame
from .scheduler import choose_interpolation_plan, output_frame_count


@dataclass(slots=True)
class ImageInterpolationResult:
    frames: np.ndarray
    report: dict[str, object]


class NearestTimestampCollector:
    """Select the output timeline without encoding a temporary video."""

    def __init__(self, target_rate: Fraction, output_count: int) -> None:
        self.target_rate = target_rate
        self.output_count = output_count
        self.next_index = 0
        self.previous: TimedFrame | None = None
        self.tie_late = False
        self.frames: list[TimedFrame] = []
        self.copied = 0
        self.generated = 0
        self.max_error = Fraction(0)
        self.selected_real_ids: set[int] = set()

    def _select(self, frame: TimedFrame, ideal: Fraction) -> None:
        self.frames.append(frame)
        self.max_error = max(self.max_error, abs(frame.timestamp - ideal))
        if frame.provenance == "DLSSG":
            self.generated += 1
        else:
            self.copied += 1
            if frame.source_index is not None:
                self.selected_real_ids.add(frame.source_index)
        self.next_index += 1

    def push(self, current: TimedFrame) -> None:
        if self.previous is None:
            self.previous = current
            return
        midpoint = (self.previous.timestamp + current.timestamp) / 2
        while self.next_index < self.output_count:
            ideal = Fraction(self.next_index, 1) / self.target_rate
            if ideal < midpoint:
                self._select(self.previous, ideal)
            elif ideal == midpoint:
                selected = current if self.tie_late else self.previous
                self.tie_late = not self.tie_late
                self._select(selected, ideal)
            else:
                break
        self.previous = current

    def finish(self) -> None:
        if self.previous is None:
            raise ValueError("Frame interpolation requires at least two IMAGE frames.")
        while self.next_index < self.output_count:
            ideal = Fraction(self.next_index, 1) / self.target_rate
            self._select(self.previous, ideal)


def interpolate_image_sequence(
    frames: np.ndarray,
    input_rate: Fraction,
    options: FrameInterpolationOptions,
    progress: Callable[[float, str], None] | None = None,
) -> ImageInterpolationResult:
    """Interpolate contiguous RGBA frames and return the selected IMAGE timeline."""
    if frames.ndim != 4 or frames.shape[-1] != 4:
        raise ValueError("IMAGE interpolation requires contiguous RGBA frames.")
    if len(frames) < 2:
        raise ValueError("Frame interpolation requires at least two IMAGE frames.")
    if input_rate <= 0:
        raise ValueError("Input FPS must be positive.")

    prepared_runtime = prepare_runtime()
    ai_gpu = resolve_runtime_ai_gpu(
        prepared_runtime.gpus, prepared_runtime.runtime_bundle, options.ai_gpu_uuid
    )
    capabilities = probe_frame_interpolation_capabilities(options.ai_gpu_uuid)
    if not capabilities.available:
        raise RuntimeError(
            "Direct NVIDIA DLSS Frame Generation is unavailable. "
            + capabilities.detail
        )
    plan = choose_interpolation_plan(
        input_rate,
        options.target_rate,
        options.engine,
        capabilities.native_multiplier,
        cfr=True,
    )
    duration = Fraction(len(frames), 1) / input_rate
    output_count = output_frame_count(duration, options.target_rate)
    if output_count < 1:
        raise ValueError("Requested output FPS produces no output IMAGE frames.")

    with active_job() as controller:
        assert controller is not None
        started = time.perf_counter()
        sessions: list[DirectDLSSGSession] = []
        result: ImageInterpolationResult | None = None
        close_error: Exception | None = None
        try:
            height, width = map(int, frames.shape[1:3])
            if plan.path == "Native DLSSG":
                sessions.append(
                    DirectDLSSGSession(
                        width,
                        height,
                        len(frames),
                        plan.generated_per_interval,
                        controller,
                    )
                )
            elif plan.path == "Cascade":
                for stage_index in range(plan.cascade_stages):
                    expected = max(1, (len(frames) - 1) * (1 << stage_index) + 1)
                    sessions.append(
                        DirectDLSSGSession(width, height, expected, 1, controller)
                    )
            stages = [
                DLSSGStage(
                    session,
                    width,
                    height,
                    plan.generated_per_interval if plan.path == "Native DLSSG" else 1,
                    detect_source_cuts=index == 0,
                )
                for index, session in enumerate(sessions)
            ]
            collector = NearestTimestampCollector(options.target_rate, output_count)
            for index, rgba in enumerate(frames):
                if controller.cancel.is_set():
                    raise Cancelled("Frame interpolation was cancelled.")
                items = [
                    TimedFrame(
                        np.ascontiguousarray(rgba, dtype=np.uint8),
                        Fraction(index, 1) / input_rate,
                        0,
                        "Source",
                        index,
                    )
                ]
                for stage in stages:
                    next_items: list[TimedFrame] = []
                    for item in items:
                        next_items.extend(stage.push(item))
                    items = next_items
                for item in items:
                    collector.push(item)
                if progress is not None:
                    progress((index + 1) / len(frames), "Interpolating IMAGE sequence")
            collector.finish()
            scene_cuts = sum(stage.scene_cuts for stage in stages)
            duplicates = sum(stage.duplicates for stage in stages)
            result_frames = np.stack([frame.rgba for frame in collector.frames])
            elapsed = time.perf_counter() - started
            report = {
                "status": "success",
                "source_type": "image_sequence",
                "input_fps": str(input_rate),
                "output_fps": str(options.target_rate),
                "input_frames": len(frames),
                "output_frames": output_count,
                "copied_frames": collector.copied,
                "generated_frames": collector.generated,
                "dropped_frames": max(0, len(frames) - len(collector.selected_real_ids)),
                "maximum_temporal_approximation_seconds": float(collector.max_error),
                "scene_cuts": scene_cuts,
                "duplicate_intervals": duplicates,
                "plan": {
                    key: str(value) if isinstance(value, Fraction) else value
                    for key, value in asdict(plan).items()
                },
                "capabilities": asdict(capabilities),
                "gpu": ai_gpu,
                "elapsed_seconds": elapsed,
            }
            result = ImageInterpolationResult(result_frames, report)
        except Exception:
            for session in sessions:
                if session is not None:
                    with suppress(Exception):
                        session.abort()
            raise
        finally:
            for session in sessions:
                try:
                    session.close()
                except Exception as exc:
                    if close_error is None:
                        close_error = exc
    if close_error is not None and result is not None:
        raise close_error
    return result
