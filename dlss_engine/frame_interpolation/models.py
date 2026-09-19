from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction


FPS_RATES: dict[str, Fraction] = {
    "23.976": Fraction(24000, 1001),
    "24": Fraction(24, 1),
    "25": Fraction(25, 1),
    "29.97": Fraction(30000, 1001),
    "30": Fraction(30, 1),
    "50": Fraction(50, 1),
    "59.94": Fraction(60000, 1001),
    "60": Fraction(60, 1),
    "90": Fraction(90, 1),
    "120": Fraction(120, 1),
}
FPS_CHOICES = tuple(FPS_RATES)
SOURCE_FPS_RATES: dict[str, Fraction] = {
    "8": Fraction(8, 1),
    "12": Fraction(12, 1),
    "16": Fraction(16, 1),
    "20": Fraction(20, 1),
    **FPS_RATES,
}
SOURCE_FPS_CHOICES = tuple(SOURCE_FPS_RATES)
ENGINE_CHOICES = ("Auto", "Native DLSSG", "Cascade")


def resolve_target_rate(value: str | int | float | Fraction) -> Fraction:
    return _resolve_rate(value, FPS_RATES, "Output FPS")


def resolve_source_rate(value: str | int | float | Fraction) -> Fraction:
    return _resolve_rate(value, SOURCE_FPS_RATES, "Input FPS")


def _resolve_rate(
    value: str | int | float | Fraction,
    rates: dict[str, Fraction],
    label: str,
) -> Fraction:
    if isinstance(value, Fraction):
        rate = value
    else:
        key = str(value).strip()
        try:
            rate = rates[key]
        except KeyError as exc:
            choices = ", ".join(rates)
            raise ValueError(f"Unsupported {label} {value!r}. Choose one of: {choices}.") from exc
    if rate <= 0:
        raise ValueError(f"{label} must be positive.")
    return rate


@dataclass(slots=True)
class FrameInterpolationOptions:
    ai_gpu_uuid: str = "auto"
    video_gpu_uuid: str = "auto"
    target_fps: str = "60"
    engine: str = "Auto"
    codec: str = "H.264"
    container: str = "MP4"
    quality: str = "Auto (Default)"
    hdr_mode: bool = False
    rename_mode: str = "Auto"
    custom_suffix: str = "_DLSSFG"
    preview_seconds: float | None = None
    # True = truncated preview uses the forced H.264 SDR path (current behavior).
    # False = truncated preview uses the user's codec/container (HDR preserved).
    preview_compat: bool = True

    @property
    def target_rate(self) -> Fraction:
        return resolve_target_rate(self.target_fps)


@dataclass(slots=True)
class FrameInterpolationCapabilities:
    available: bool
    gpu: str
    driver: str
    hags_enabled: bool
    native_generated_frame_max: int
    native_multiplier: int
    cascade_available: bool
    runtime_version: str
    worker_version: str
    signature_status: str
    detail: str = ""
    gpu_uuid: str = "auto"


@dataclass(frozen=True, slots=True)
class InterpolationPlan:
    path: str
    source_rate: Fraction
    target_rate: Fraction
    native_multiplier: int
    grid_multiplier: int
    cascade_stages: int
    maximum_temporal_error: Fraction
    generated_per_interval: int


@dataclass(slots=True)
class FrameInterpolationResult:
    output_path: str
    report_path: str
    selected_path: str
    native_multiplier: int
    cascade_stages: int
    copied_frames: int
    generated_frames: int
    dropped_frames: int
    output_frames: int
    maximum_temporal_approximation_seconds: float
    scene_cuts: int
    elapsed_seconds: float
    timings: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class FrameInterpolationSuccess:
    index: int
    input_path: str
    result: FrameInterpolationResult


@dataclass(slots=True)
class FrameInterpolationFailure:
    index: int
    input_path: str
    error: str
    cancelled: bool = False


@dataclass(slots=True)
class FrameInterpolationBatchResult:
    successes: list[FrameInterpolationSuccess]
    failures: list[FrameInterpolationFailure]
    cancelled: bool
    manifest_path: str
