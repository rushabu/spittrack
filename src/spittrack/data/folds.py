"""Deterministic, balanced, source-grouped fold construction."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from spittrack.common.errors import FoldError
from spittrack.data.contracts import SourceRecord


@dataclass(frozen=True)
class FoldManifest:
    """One leakage-safe outer fold with a source-grouped validation partition."""

    fold: int
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    external_real: tuple[str, ...]
    seed: int

    def validate(self) -> None:
        partitions = {
            "train": set(self.train),
            "validation": set(self.validation),
            "test": set(self.test),
            "external_real": set(self.external_real),
        }
        names = tuple(partitions)
        for index, left_name in enumerate(names):
            for right_name in names[index + 1 :]:
                overlap = partitions[left_name] & partitions[right_name]
                if overlap:
                    raise FoldError(
                        f"fold {self.fold}: source leakage between {left_name} and {right_name}: "
                        f"{sorted(overlap)}"
                    )

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": 1,
            "fold": self.fold,
            "seed": self.seed,
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
            "external_real": list(self.external_real),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return payload


def _round_robin_chunks(values: list[str], k: int) -> list[list[str]]:
    chunks = [[] for _ in range(k)]
    for index, value in enumerate(values):
        chunks[index % k].append(value)
    return chunks


def build_balanced_folds(
    sources: Iterable[SourceRecord],
    *,
    k: int = 5,
    seed: int = 42,
    validation_fraction: float = 0.25,
) -> list[FoldManifest]:
    """Build deterministic folds or fail instead of reducing K or weakening balance."""
    if k < 2:
        raise FoldError(f"K must be at least 2, got {k}")
    if not 0.0 < validation_fraction < 1.0:
        raise FoldError("validation_fraction must be strictly between 0 and 1")

    source_list = list(sources)
    synthetic = [source for source in source_list if source.domain == "synthetic"]
    real = tuple(sorted(source.source_id for source in source_list if source.domain == "real_staged"))
    positives = sorted(source.source_id for source in synthetic if source.contains_spit)
    negatives = sorted(source.source_id for source in synthetic if not source.contains_spit)
    required_total = 2 * k
    if len(positives) < k or len(negatives) < k:
        raise FoldError(
            "Balanced source-grouped K-fold construction is impossible: "
            f"K={k} requires N >= {required_total}, with at least {k} contains_spit=true and "
            f"{k} contains_spit=false sources; found N={len(synthetic)} "
            f"({len(positives)} positive, {len(negatives)} negative). K will not be reduced."
        )
    if len(positives) != len(negatives):
        raise FoldError(
            "Exactly balanced folds require equal synthetic class counts; "
            f"found {len(positives)} positive and {len(negatives)} negative. "
            "No source will be dropped and folds will not be unbalanced silently."
        )

    positive_rng = random.Random(seed)
    negative_rng = random.Random(seed + 1)
    positive_rng.shuffle(positives)
    negative_rng.shuffle(negatives)
    positive_tests = _round_robin_chunks(positives, k)
    negative_tests = _round_robin_chunks(negatives, k)

    all_synthetic_ids = set(positives) | set(negatives)
    folds: list[FoldManifest] = []
    for fold_index in range(k):
        test_pos = positive_tests[fold_index]
        test_neg = negative_tests[fold_index]
        if len(test_pos) != len(test_neg):
            raise FoldError(
                f"fold {fold_index + 1}: test partition would be unbalanced "
                f"({len(test_pos)} positive, {len(test_neg)} negative)"
            )
        remaining_pos = [value for value in positives if value not in test_pos]
        remaining_neg = [value for value in negatives if value not in test_neg]
        if len(remaining_pos) < 2 or len(remaining_neg) < 2:
            raise FoldError(
                f"fold {fold_index + 1}: insufficient sources remain for both balanced validation "
                "and training partitions"
            )
        validation_per_class = max(1, round(len(remaining_pos) * validation_fraction))
        validation_per_class = min(validation_per_class, len(remaining_pos) - 1)
        rotation = fold_index % len(remaining_pos)
        rotated_pos = remaining_pos[rotation:] + remaining_pos[:rotation]
        rotated_neg = remaining_neg[rotation:] + remaining_neg[:rotation]
        validation = tuple(sorted(rotated_pos[:validation_per_class] + rotated_neg[:validation_per_class]))
        test = tuple(sorted(test_pos + test_neg))
        train = tuple(sorted(all_synthetic_ids - set(validation) - set(test)))
        manifest = FoldManifest(
            fold=fold_index + 1,
            train=train,
            validation=validation,
            test=test,
            external_real=real,
            seed=seed,
        )
        manifest.validate()
        folds.append(manifest)

    tested = [source_id for fold in folds for source_id in fold.test]
    if sorted(tested) != sorted(all_synthetic_ids) or len(tested) != len(set(tested)):
        raise FoldError("Outer-test coverage invariant failed: every synthetic source must appear exactly once")
    return folds


def write_fold_manifests(
    folds: Iterable[FoldManifest], output_dir: Path, *, overwrite: bool = False
) -> list[Path]:
    """Write immutable JSON and text manifests, refusing accidental replacement by default."""
    fold_list = list(folds)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets: list[Path] = []
    for fold in fold_list:
        prefix = f"fold_{fold.fold:02d}"
        targets.extend(
            [
                output_dir / f"{prefix}.json",
                output_dir / f"{prefix}_train.txt",
                output_dir / f"{prefix}_val.txt",
                output_dir / f"{prefix}_test.txt",
            ]
        )
    targets.append(output_dir / "external_real.txt")
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FoldError(
            "Fold manifests are immutable inputs and already exist; refusing to overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    written: list[Path] = []
    for fold in fold_list:
        prefix = f"fold_{fold.fold:02d}"
        json_path = output_dir / f"{prefix}.json"
        json_path.write_text(json.dumps(fold.as_dict(), indent=2) + "\n", encoding="utf-8")
        written.append(json_path)
        for suffix, values in (("train", fold.train), ("val", fold.validation), ("test", fold.test)):
            path = output_dir / f"{prefix}_{suffix}.txt"
            path.write_text("\n".join(values) + "\n", encoding="utf-8")
            written.append(path)
    external = fold_list[0].external_real if fold_list else ()
    external_path = output_dir / "external_real.txt"
    external_path.write_text("\n".join(external) + ("\n" if external else ""), encoding="utf-8")
    written.append(external_path)
    return written


def load_fold_manifest(path: Path) -> FoldManifest:
    """Load an immutable manifest and reject content whose recorded hash no longer matches."""
    if not path.is_file():
        raise FoldError(f"Fold manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FoldError(f"Fold manifest is unreadable: {path}: {exc}") from exc
    required = {
        "schema_version",
        "fold",
        "seed",
        "train",
        "validation",
        "test",
        "external_real",
        "manifest_sha256",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise FoldError(f"Fold manifest is missing fields {missing}: {path}")
    recorded_hash = payload.pop("manifest_sha256")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    actual_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if recorded_hash != actual_hash:
        raise FoldError(
            f"Fold manifest hash mismatch: {path}; recorded={recorded_hash}, actual={actual_hash}"
        )
    if payload["schema_version"] != 1:
        raise FoldError(f"Unsupported fold manifest schema_version={payload['schema_version']}")
    try:
        manifest = FoldManifest(
            fold=int(payload["fold"]),
            train=tuple(str(value) for value in payload["train"]),
            validation=tuple(str(value) for value in payload["validation"]),
            test=tuple(str(value) for value in payload["test"]),
            external_real=tuple(str(value) for value in payload["external_real"]),
            seed=int(payload["seed"]),
        )
    except (TypeError, ValueError) as exc:
        raise FoldError(f"Fold manifest fields have invalid types: {path}") from exc
    manifest.validate()
    return manifest
