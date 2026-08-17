from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from spittrack.data.materialize import materialize_fullframe_clips
from spittrack.data.windows import build_fullframe_window_index


def test_fullframe_materialization_uses_sixteen_rgb_frames(tmp_path: Path, source_factory) -> None:
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    video = normalized / "S01.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
    assert writer.isOpened()
    for index in range(50):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 2] = index
        writer.write(frame)
    writer.release()
    source = source_factory("S01", False, duration_s=2.0)
    records, _summary = build_fullframe_window_index([source], [])
    outputs = materialize_fullframe_clips(
        records,
        normalized_root=normalized,
        clips_root=tmp_path / "clips",
    )
    assert len(outputs) == 1
    clip = np.load(outputs[0], allow_pickle=False)
    assert clip.shape == (16, 224, 224, 3)
    assert clip.dtype == np.uint8
