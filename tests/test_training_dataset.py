from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from spittrack.data.folds import FoldManifest
from spittrack.training.dataset import WindowClipDataset, source_class_balanced_sampler


def test_dataset_joins_manifest_and_excludes_ignore(tmp_path: Path) -> None:
    clips = tmp_path / "clips" / "S01"
    clips.mkdir(parents=True)
    array = np.zeros((16, 224, 224, 3), dtype=np.uint8)
    np.save(clips / "w1.npy", array, allow_pickle=False)
    np.save(clips / "w2.npy", array, allow_pickle=False)
    index = tmp_path / "windows.csv"
    with index.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["window_id", "source_id", "track_id", "label", "clip_path"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "window_id": "w1",
                "source_id": "S01",
                "track_id": 1,
                "label": "spitting",
                "clip_path": "clips/S01/w1.npy",
            }
        )
        writer.writerow(
            {
                "window_id": "w2",
                "source_id": "S01",
                "track_id": 1,
                "label": "ignore",
                "clip_path": "clips/S01/w2.npy",
            }
        )
    manifest = FoldManifest(
        fold=1,
        train=("S01",),
        validation=("S02",),
        test=("S03",),
        external_real=(),
        seed=42,
    )
    dataset = WindowClipDataset(index, tmp_path, manifest, "train")
    assert len(dataset) == 1
    clip, label, metadata = dataset[0]
    assert clip.shape == (16, 3, 224, 224)
    assert label.item() == 1
    assert metadata["window_id"] == "w1"


def test_source_class_sampler_requires_both_classes(tmp_path: Path) -> None:
    clips = tmp_path / "clips" / "S01"
    clips.mkdir(parents=True)
    np.save(clips / "w1.npy", np.zeros((16, 224, 224, 3), dtype=np.uint8), allow_pickle=False)
    index = tmp_path / "windows.csv"
    index.write_text(
        "window_id,source_id,track_id,label,clip_path\n"
        "w1,S01,1,spitting,clips/S01/w1.npy\n",
        encoding="utf-8",
    )
    manifest = FoldManifest(1, ("S01",), ("S02",), ("S03",), (), 42)
    dataset = WindowClipDataset(index, tmp_path, manifest, "train")
    import pytest

    with pytest.raises(ValueError, match="both classes"):
        source_class_balanced_sampler(dataset)

