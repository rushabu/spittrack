"""Playback-duration-preserving constant-frame-rate video normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from spittrack.common.errors import ContractError


@dataclass(frozen=True)
class NormalizationResult:
    input_fps: float
    output_fps: float
    input_frames: int
    output_frames: int
    width: int
    height: int
    input_duration_s: float
    output_duration_s: float


def normalize_video(input_path: Path, output_path: Path, *, target_fps: int = 25) -> NormalizationResult:
    """Resample frames to target FPS while preserving duration within one output frame."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError("opencv-python-headless is required for video normalization") from exc
    if target_fps <= 0:
        raise ContractError("target_fps must be positive")
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ContractError(f"Could not open input video: {input_path}")
    try:
        input_fps = float(capture.get(cv2.CAP_PROP_FPS))
        input_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if input_fps <= 0 or input_frames <= 0 or width <= 0 or height <= 0:
            raise ContractError(f"Input video reports invalid metadata: {input_path}")
        input_duration = input_frames / input_fps
        output_frames = max(1, round(input_duration * target_fps))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(target_fps),
            (width, height),
        )
        if not writer.isOpened():
            raise ContractError(f"Could not create normalized video: {output_path}")
        try:
            current_index = -1
            current_frame = None
            for output_index in range(output_frames):
                desired_index = min(
                    input_frames - 1,
                    round(output_index * input_fps / target_fps),
                )
                while current_index < desired_index:
                    ok, frame = capture.read()
                    if not ok:
                        raise ContractError(
                            f"Input video ended unexpectedly at frame {current_index + 1}: {input_path}"
                        )
                    current_index += 1
                    current_frame = frame
                if current_frame is None:
                    ok, current_frame = capture.read()
                    if not ok:
                        raise ContractError(f"Input video contains no decodable frames: {input_path}")
                    current_index = 0
                writer.write(current_frame)
        finally:
            writer.release()
    finally:
        capture.release()
    output_duration = output_frames / target_fps
    if abs(output_duration - input_duration) > (1.0 / target_fps + 1e-9):
        raise ContractError(
            f"Normalization changed duration beyond one-frame tolerance: "
            f"input={input_duration:.6f}s output={output_duration:.6f}s"
        )
    return NormalizationResult(
        input_fps=input_fps,
        output_fps=float(target_fps),
        input_frames=input_frames,
        output_frames=output_frames,
        width=width,
        height=height,
        input_duration_s=input_duration,
        output_duration_s=output_duration,
    )

