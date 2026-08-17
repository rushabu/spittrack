"""Validate source metadata and action annotations without mutating data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spittrack.common.errors import ContractError
from spittrack.data.contracts import load_action_annotations, load_source_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/source_metadata.csv"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/action_intervals.csv"))
    parser.add_argument("--synthetic-root", type=Path, default=Path("data/raw/synthetic"))
    parser.add_argument("--real-root", type=Path, default=Path("data/raw/real_staged"))
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Allow header-only CSV templates and skip referenced-file checks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        sources = load_source_metadata(
            args.metadata,
            {"synthetic": args.synthetic_root, "real_staged": args.real_root},
            require_files=not args.schema_only,
            allow_empty=args.schema_only,
        )
        annotations = load_action_annotations(
            args.annotations,
            sources,
            allow_empty=args.schema_only,
        )
    except ContractError as exc:
        logging.error("Data contract validation failed: %s", exc)
        return 2
    logging.info(
        "Validated %d source rows and %d action intervals%s",
        len(sources),
        len(annotations),
        " (schema-only mode)" if args.schema_only else "",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
