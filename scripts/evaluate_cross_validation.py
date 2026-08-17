"""Aggregate a merged model/fold output tree only when the expected matrix is complete."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spittrack.common.errors import ContractError
from spittrack.common.hashing import file_sha256
from spittrack.evaluation.cross_validation import (
    aggregate_complete_matrix,
    select_final_model_policy,
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
    parser.add_argument(
        "--selection-models",
        nargs="+",
        default=["resnet50", "resnet_lstm", "videomae"],
        help="Track-wise candidates eligible for the final person-centric model.",
    )
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
        policy = select_final_model_policy(
            args.output_root,
            candidate_models=args.selection_models,
            folds=range(1, args.folds + 1),
        )
        policy["cv_summary_sha256"] = file_sha256(destination)
        policy_path = args.output_root / "cross_validation" / "final_model_policy.json"
        policy_path.write_text(
            json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except ContractError as exc:
        logging.error("CV aggregation failed: %s", exc)
        return 2
    logging.info("Wrote complete-matrix CV summary: %s", destination)
    logging.info("Froze final-model policy: %s", policy_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
