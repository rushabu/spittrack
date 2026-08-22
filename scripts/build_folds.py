"""Generate deterministic balanced fold manifests exactly once."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import logging
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spittrack.common.errors import ContractError, FoldError
from spittrack.data.contracts import load_source_metadata
from spittrack.data.folds import build_balanced_folds, write_fold_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/source_metadata.csv"))
    parser.add_argument("--synthetic-root", type=Path, default=Path("data/raw/synthetic"))
    parser.add_argument("--real-root", type=Path, default=Path("data/raw/real_staged"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument(
        "--window-index",
        type=Path,
        default=Path("data/processed/windows/windows.csv"),
        help="Optional fold-independent window index used only for label-count reporting.",
    )
    return parser.parse_args()


def _window_counts_by_source(path: Path) -> dict[str, Counter[str]] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ContractError(f"Window index has no header: {path}")
        missing = sorted({"source_id", "label"} - set(reader.fieldnames))
        if missing:
            raise ContractError(f"Window index is missing columns {missing}: {path}")
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for row_number, row in enumerate(reader, start=2):
            source_id = (row.get("source_id") or "").strip()
            label = (row.get("label") or "").strip()
            if not source_id:
                raise ContractError(f"Window index row {row_number}: source_id cannot be blank")
            if label not in {"spitting", "non_spitting", "ignore"}:
                raise ContractError(
                    f"Window index row {row_number}: unknown label {label!r}"
                )
            counts[source_id][label] += 1
    return counts


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        sources = load_source_metadata(
            args.metadata,
            {"synthetic": args.synthetic_root, "real_staged": args.real_root},
        )
        folds = build_balanced_folds(
            sources,
            k=args.k,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
        )
        window_counts = _window_counts_by_source(args.window_index)
        paths = write_fold_manifests(folds, args.output_dir)
    except (ContractError, FoldError) as exc:
        logging.error("Fold construction failed: %s", exc)
        return 2
    logging.info("Wrote %d immutable manifest files for %d folds", len(paths), len(folds))
    if window_counts is None:
        logging.warning(
            "Window index not found at %s; per-fold window-label counts are unavailable. "
            "Build the fold-independent window index first when these counts are required.",
            args.window_index,
        )
    class_by_family = {
        source.family_id: source.contains_spit
        for source in sources
        if source.domain == "synthetic" and source.family_id is not None
    }
    positive_family_counts = [
        sum(class_by_family[family_id] for family_id in fold.test_families)
        for fold in folds
    ]
    mean_positive_families = sum(positive_family_counts) / len(positive_family_counts)
    for fold in folds:
        fold_positive_families = sum(
            class_by_family[family_id] for family_id in fold.test_families
        )
        fold_negative_families = len(fold.test_families) - fold_positive_families
        if window_counts is None:
            label_report = "positive=NA negative=NA ignore=NA"
        else:
            labels = Counter()
            for source_id in fold.test:
                labels.update(window_counts.get(source_id, Counter()))
            label_report = (
                f"positive={labels['spitting']} negative={labels['non_spitting']} "
                f"ignore={labels['ignore']}"
            )
        logging.info(
            "fold=%02d outer_test=%d families (%d positive/%d negative), %d sources, "
            "windows (%s); train=%d sources/%d families; "
            "validation=%d sources/%d families; external_real=%d sources",
            fold.fold,
            len(fold.test_families),
            fold_positive_families,
            fold_negative_families,
            len(fold.test),
            label_report,
            len(fold.train),
            len(fold.train_families),
            len(fold.validation),
            len(fold.validation_families),
            len(fold.external_real),
        )
        if abs(fold_positive_families - mean_positive_families) > 1:
            logging.warning(
                "fold=%02d positive-family count=%d deviates from mean %.3f by more than one",
                fold.fold,
                fold_positive_families,
                mean_positive_families,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
