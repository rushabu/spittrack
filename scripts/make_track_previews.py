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
        "--real-staged",
        action="store_true",
        help="Use isolated staged-real normalized videos, manifests, and preview outputs.",
    )
    parser.add_argument(
        "--normalized-root",
        type=Path,
        help="Override the domain-specific normalized-video root.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        help="Override the domain-specific tracking-manifest directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the domain-specific preview output directory.",
    )
    parser.add_argument("--source-id", help="Render only one source")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        normalized_root = args.normalized_root or Path(
            "data/normalized/real_staged"
            if args.real_staged
            else "data/normalized/synthetic"
        )
        manifest_dir = args.manifest_dir or Path(
            "data/tracking_real/manifests"
            if args.real_staged
            else "data/tracking/manifests"
        )
        output_dir = args.output_dir or Path(
            "data/tracking_real/previews"
            if args.real_staged
            else "data/tracking/previews"
        )
        if args.real_staged:
            synthetic_tracking = Path("data/tracking").resolve()
            resolved_output = output_dir.resolve()
            if resolved_output == synthetic_tracking or synthetic_tracking in resolved_output.parents:
                raise ContractError(
                    "Real preview rendering refuses to write under the synthetic tracking tree"
                )
        if not manifest_dir.is_dir():
            raise ContractError(
                f"Tracking manifest directory does not exist: {manifest_dir}"
            )
        manifests = sorted(manifest_dir.glob("*.csv"))
        if args.source_id:
            manifests = [path for path in manifests if path.stem == args.source_id]
            if not manifests:
                raise ContractError(f"Unknown tracked --source-id {args.source_id!r}")
        if not manifests:
            raise ContractError(f"No tracking manifest CSV files found in: {manifest_dir}")

        for manifest_path in manifests:
            source_id = manifest_path.stem
            output_path = output_dir / f"{source_id}.mp4"
            frame_count = render_track_preview(
                normalized_root / f"{source_id}.mp4",
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
