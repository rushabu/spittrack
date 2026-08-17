from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from spittrack.data.normalize import normalize_video


def test_normalization_preserves_duration_within_one_output_frame(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (64, 48))
    assert writer.isOpened()
    for index in range(40):
        writer.write(np.full((48, 64, 3), index, dtype=np.uint8))
    writer.release()
    output = tmp_path / "normalized.mp4"
    result = normalize_video(source, output, target_fps=25)
    assert output.is_file()
    assert result.input_frames == 40
    assert result.output_frames == 50
    assert abs(result.input_duration_s - result.output_duration_s) <= 1 / 25

