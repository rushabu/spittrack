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
        assert not (set(fold.train_families) & set(fold.validation_families))
        assert not (set(fold.train_families) & set(fold.test_families))
        assert not (set(fold.validation_families) & set(fold.test_families))

    assert sorted(tested) == [f"S{index:02d}" for index in range(1, 21)]
    assert len(tested) == len(set(tested))


def test_all_variants_of_a_family_stay_in_one_partition_and_each_family_is_tested_once(
    source_factory,
) -> None:
    sources = []
    family_by_source: dict[str, str] = {}
    for family_index in range(1, 21):
        family_id = f"shot-family-{family_index}"
        variant_count = 2 if family_index <= 17 else 1  # 37 source videos, 20 families
        for variant_index in range(variant_count):
            source_id = f"generated-clip-{family_index}-{variant_index}"
            sources.append(
                source_factory(
                    source_id,
                    family_index <= 10,
                    family_id=family_id,
                )
            )
            family_by_source[source_id] = family_id

    folds = build_balanced_folds(sources, k=5, seed=42)
    tested_families: list[str] = []
    for fold in folds:
        family_partitions = {
            "train": set(fold.train_families),
            "validation": set(fold.validation_families),
            "test": set(fold.test_families),
        }
        source_partitions = {
            "train": set(fold.train),
            "validation": set(fold.validation),
            "test": set(fold.test),
        }
        for partition, source_ids in source_partitions.items():
            assert {family_by_source[source_id] for source_id in source_ids} == family_partitions[
                partition
            ]
        tested_families.extend(fold.test_families)

    assert len(sources) == 37
    assert sorted(tested_families) == sorted(
        f"shot-family-{index}" for index in range(1, 21)
    )
    assert len(tested_families) == len(set(tested_families))


def test_impossible_five_fold_reports_required_minimum(source_factory) -> None:
    sources = [source_factory(f"P{index}", True) for index in range(4)]
    sources += [source_factory(f"N{index}", False) for index in range(4)]
    with pytest.raises(
        FoldError,
        match=r"at least 10 synthetic families.*at least 5 contains_spit=true families",
    ):
        build_balanced_folds(sources, k=5)


def test_unequal_classes_are_not_silently_unbalanced(source_factory) -> None:
    sources = [source_factory(f"P{index}", True) for index in range(6)]
    sources += [source_factory(f"N{index}", False) for index in range(5)]
    with pytest.raises(FoldError, match="equal positive and negative family counts"):
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
