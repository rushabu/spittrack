"""Pure bounding-box scaling, padding, and clamping utilities."""

from __future__ import annotations

from spittrack.common.errors import ContractError


def padded_clamped_box(
    box_xyxy: tuple[float, float, float, float],
    *,
    frame_width: int,
    frame_height: int,
    padding_fraction: float = 0.18,
) -> tuple[int, int, int, int]:
    """Expand each side by a fraction of box size and clamp to valid pixel bounds."""
    if frame_width <= 0 or frame_height <= 0:
        raise ContractError("Frame dimensions must be positive")
    if padding_fraction < 0:
        raise ContractError("padding_fraction cannot be negative")
    x1, y1, x2, y2 = box_xyxy
    if x2 <= x1 or y2 <= y1:
        raise ContractError(f"Cannot pad zero-size or inverted box {box_xyxy}")
    pad_x = (x2 - x1) * padding_fraction
    pad_y = (y2 - y1) * padding_fraction
    left = max(0, min(frame_width, int((x1 - pad_x) // 1)))
    top = max(0, min(frame_height, int((y1 - pad_y) // 1)))
    right = max(0, min(frame_width, int(-(-(x2 + pad_x) // 1))))
    bottom = max(0, min(frame_height, int(-(-(y2 + pad_y) // 1))))
    if right <= left or bottom <= top:
        raise ContractError(
            f"Padded box became invalid after clamping: {box_xyxy} in {frame_width}x{frame_height}"
        )
    return left, top, right, bottom


def bbox_height_at_720p(box_height: float, frame_height: int) -> float:
    """Scale a box height to a 720-pixel-tall reference frame."""
    if box_height <= 0 or frame_height <= 0:
        raise ContractError("Box and frame heights must be positive")
    return box_height * 720.0 / frame_height

