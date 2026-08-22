"""Generate deterministic balanced fold manifests exactly once."""

from __future__ import annotations

import argparse
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
    return parser.parse_args()


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
        paths = write_fold_manifests(folds, args.output_dir)
    except (ContractError, FoldError) as exc:
        logging.error("Fold construction failed: %s", exc)
        return 2
    logging.info("Wrote %d immutable manifest files for %d folds", len(paths), len(folds))
    for fold in folds:
        logging.info(
            "fold=%02d train=%d sources/%d families validation=%d sources/%d families "
            "test=%d sources/%d families external_real=%d sources",
            fold.fold,
            len(fold.train),
            len(fold.train_families),
            len(fold.validation),
            len(fold.validation_families),
            len(fold.test),
            len(fold.test_families),
            len(fold.external_real),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
