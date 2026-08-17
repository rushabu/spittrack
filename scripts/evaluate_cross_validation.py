"""Aggregate a merged model/fold output tree only when the expected matrix is complete."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spittrack.common.errors import ContractError
from spittrack.evaluation.cross_validation import (
    aggregate_complete_matrix,
    write_cross_validation_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet50", "resnet_lstm", "videomae", "videomae_fullframe"],
    )
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        aggregates = aggregate_complete_matrix(
            args.output_root,
            models=args.models,
            folds=range(1, args.folds + 1),
        )
        destination = args.output_root / "cross_validation" / "cross_validation_summary.csv"
        write_cross_validation_summary(aggregates, destination)
    except ContractError as exc:
        logging.error("CV aggregation failed: %s", exc)
        return 2
    logging.info("Wrote complete-matrix CV summary: %s", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

