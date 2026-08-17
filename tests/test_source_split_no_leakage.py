from __future__ import annotations

from pathlib import Path

import pytest

from spittrack.common.errors import FoldError
from spittrack.data.folds import build_balanced_folds, load_fold_manifest, write_fold_manifests


def test_five_folds_are_balanced_disjoint_and_cover_every_synthetic_source_once(
    source_factory,
) -> None:
    sources = [source_factory(f"S{index:02d}", index <= 10) for index in range(1, 21)]
    sources += [source_factory("R01", True, domain="real_staged")]
    folds = build_balanced_folds(sources, k=5, seed=42)

    tested: list[str] = []
    for fold in folds:
        fold.validate()
        tested.extend(fold.test)
        positive_test = {source.source_id for source in sources if source.contains_spit} & set(fold.test)
        negative_test = {source.source_id for source in sources if not source.contains_spit} & set(fold.test)
        assert len(positive_test) == len(negative_test)
        assert "R01" not in set(fold.train) | set(fold.validation) | set(fold.test)
        assert fold.external_real == ("R01",)

    assert sorted(tested) == [f"S{index:02d}" for index in range(1, 21)]
    assert len(tested) == len(set(tested))


def test_impossible_five_fold_reports_required_minimum(source_factory) -> None:
    sources = [source_factory(f"P{index}", True) for index in range(4)]
    sources += [source_factory(f"N{index}", False) for index in range(4)]
    with pytest.raises(FoldError, match=r"N >= 10.*at least 5 contains_spit=true"):
        build_balanced_folds(sources, k=5)


def test_unequal_classes_are_not_silently_unbalanced(source_factory) -> None:
    sources = [source_factory(f"P{index}", True) for index in range(6)]
    sources += [source_factory(f"N{index}", False) for index in range(5)]
    with pytest.raises(FoldError, match="equal synthetic class counts"):
        build_balanced_folds(sources, k=5)


def test_manifests_refuse_overwrite(tmp_path: Path, source_factory) -> None:
    sources = [source_factory(f"P{index}", True) for index in range(5)]
    sources += [source_factory(f"N{index}", False) for index in range(5)]
    folds = build_balanced_folds(sources, k=5)
    write_fold_manifests(folds, tmp_path)
    with pytest.raises(FoldError, match="immutable"):
        write_fold_manifests(folds, tmp_path)


def test_manifest_hash_detects_tampering(tmp_path: Path, source_factory) -> None:
    sources = [source_factory(f"P{index}", True) for index in range(5)]
    sources += [source_factory(f"N{index}", False) for index in range(5)]
    folds = build_balanced_folds(sources, k=5)
    write_fold_manifests(folds, tmp_path)
    path = tmp_path / "fold_01.json"
    assert load_fold_manifest(path) == folds[0]
    path.write_text(path.read_text(encoding="utf-8").replace('"fold": 1', '"fold": 9'), encoding="utf-8")
    with pytest.raises(FoldError, match="hash mismatch"):
        load_fold_manifest(path)
