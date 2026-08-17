"""Guarded entry point for the one-time staged-real stress test."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


ACK_FLAG = "--i-understand-this-is-one-time"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-model-provenance", type=Path, required=True)
    parser.add_argument("--i-understand-this-is-one-time", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.i_understand_this_is_one_time:
        logging.error("Refusing staged-real evaluation without explicit %s acknowledgement", ACK_FLAG)
        return 2
    if not args.final_model_provenance.is_file():
        logging.error("Final-model provenance is missing: %s", args.final_model_provenance)
        return 2
    try:
        provenance = json.loads(args.final_model_provenance.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logging.error("Final-model provenance is unreadable: %s", exc)
        return 2
    required = {
        "model_name",
        "checkpoint_path",
        "median_cv_best_epoch",
        "median_fold_validation_threshold",
        "cv_summary_sha256",
        "config_sha256",
    }
    missing = sorted(required - set(provenance))
    if missing:
        logging.error("Final-model provenance is incomplete; missing fields: %s", missing)
        return 2
    logging.info(
        "Frozen staged-real provenance: model=%s checkpoint=%s epoch=%s threshold=%s "
        "cv_summary_sha256=%s config_sha256=%s",
        provenance["model_name"],
        provenance["checkpoint_path"],
        provenance["median_cv_best_epoch"],
        provenance["median_fold_validation_threshold"],
        provenance["cv_summary_sha256"],
        provenance["config_sha256"],
    )
    logging.error(
        "Guard passed, but staged-real inference is not implemented yet; no real clip was opened or evaluated."
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
