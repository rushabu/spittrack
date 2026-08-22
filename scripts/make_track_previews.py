"""Render normalized videos with tracked-person boxes, IDs, and frame timestamps."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spittrack.common.errors import ContractError
from spittrack.data.track_previews import render_track_preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--normalized-root",
        type=Path,
        default=Path("data/normalized/synthetic"),
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/tracking/manifests"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tracking/previews"),
    )
    parser.add_argument("--source-id", help="Render only one source")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if not args.manifest_dir.is_dir():
            raise ContractError(
                f"Tracking manifest directory does not exist: {args.manifest_dir}"
            )
        manifests = sorted(args.manifest_dir.glob("*.csv"))
        if args.source_id:
            manifests = [path for path in manifests if path.stem == args.source_id]
            if not manifests:
                raise ContractError(f"Unknown tracked --source-id {args.source_id!r}")
        if not manifests:
            raise ContractError(f"No tracking manifest CSV files found in: {args.manifest_dir}")

        for manifest_path in manifests:
            source_id = manifest_path.stem
            output_path = args.output_dir / f"{source_id}.mp4"
            frame_count = render_track_preview(
                args.normalized_root / f"{source_id}.mp4",
                manifest_path,
                output_path,
                source_id=source_id,
                overwrite=args.overwrite,
            )
            logging.info(
                "source=%s preview=%s frames=%d device=cpu",
                source_id,
                output_path,
                frame_count,
            )
    except (ContractError, OSError, ValueError) as exc:
        logging.error("Track-preview rendering failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
