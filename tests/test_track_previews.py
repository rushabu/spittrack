from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from spittrack.data.track_previews import render_track_preview
from spittrack.data.tracking import TRACKING_REQUIRED_COLUMNS


def test_track_preview_smoke_renders_boxes_ids_and_timestamp(tmp_path: Path) -> None:
    video_path = tmp_path / "arbitrary-source-name.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (96, 64)
    )
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.zeros((64, 96, 3), dtype=np.uint8))
    writer.release()

    manifest_path = tmp_path / "arbitrary-source-name.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        manifest_writer = csv.DictWriter(stream, fieldnames=TRACKING_REQUIRED_COLUMNS)
        manifest_writer.writeheader()
        for frame_index in (1, 2):
            manifest_writer.writerow(
                {
                    "source_id": "arbitrary-source-name",
                    "tracking_run_id": "smoke-run",
                    "frame_index": frame_index,
                    "timestamp_s": frame_index / 25,
                    "track_id": 7,
                    "x1": 10,
                    "y1": 30,
                    "x2": 60,
                    "y2": 55,
                    "det_conf": 0.9,
                }
            )

    output_path = tmp_path / "preview.mp4"
    assert (
        render_track_preview(
            video_path,
            manifest_path,
            output_path,
            source_id="arbitrary-source-name",
        )
        == 5
    )

    capture = cv2.VideoCapture(str(output_path))
    decoded = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(frame)
    capture.release()
    assert len(decoded) == 5
    assert np.count_nonzero(decoded[0][2:28, 2:94]) > 0  # timestamp
    assert np.count_nonzero(decoded[1][28:58, 8:63]) > np.count_nonzero(
        decoded[0][28:58, 8:63]
    )  # tracked box and ID
