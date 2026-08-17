from __future__ import annotations

import pytest

from spittrack.common.errors import ContractError
from spittrack.data.boxes import bbox_height_at_720p, padded_clamped_box


def test_eighteen_percent_expansion() -> None:
    assert padded_clamped_box(
        (100, 100, 200, 300), frame_width=640, frame_height=480, padding_fraction=0.18
    ) == (82, 64, 218, 336)


def test_padding_clamps_to_frame() -> None:
    assert padded_clamped_box(
        (5, 5, 100, 100), frame_width=100, frame_height=100, padding_fraction=0.18
    ) == (0, 0, 100, 100)


def test_zero_size_box_is_rejected() -> None:
    with pytest.raises(ContractError, match="zero-size"):
        padded_clamped_box((10, 10, 10, 20), frame_width=100, frame_height=100)


def test_bbox_height_normalizes_to_720p_reference() -> None:
    assert bbox_height_at_720p(160, 1440) == pytest.approx(80)

