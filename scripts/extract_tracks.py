"""Run frozen YOLOv8s + ByteTrack over normalized synthetic sources."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spittrack.common.config import load_yaml
from spittrack.common.errors import ContractError
from spittrack.data.contracts import load_source_metadata
from spittrack.data.extract_tracks import extract_track_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/source_metadata.csv"))
    parser.add_argument("--synthetic-root", type=Path, default=Path("data/raw/synthetic"))
    parser.add_argument("--real-root", type=Path, default=Path("data/raw/real_staged"))
    parser.add_argument("--normalized-root", type=Path, default=Path("data/normalized/synthetic"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/tracking/manifests"))
    parser.add_argument("--detector-config", type=Path, default=Path("configs/detector.yaml"))
    parser.add_argument("--tracker-config", type=Path, default=Path("configs/tracker.yaml"))
    parser.add_argument("--source-id", help="Process only one source for a smoke test")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        sources = load_source_metadata(
            args.metadata,
            {"synthetic": args.synthetic_root, "real_staged": args.real_root},
        )
        synthetic = [source for source in sources if source.domain == "synthetic"]
        if args.source_id:
            synthetic = [source for source in synthetic if source.source_id == args.source_id]
            if not synthetic:
                raise ContractError(f"Unknown synthetic --source-id {args.source_id!r}")
        detector = load_yaml(args.detector_config)
        tracker = load_yaml(args.tracker_config)
        for source in synthetic:
            video_path = args.normalized_root / f"{source.source_id}.mp4"
            if not video_path.is_file():
                raise ContractError(f"Normalized source does not exist: {video_path}")
            output_path = args.output_dir / f"{source.source_id}.csv"
            if output_path.exists():
                raise ContractError(
                    f"Tracking manifest already exists; immutable run artifacts are not overwritten: {output_path}"
                )
            extract_track_manifest(
                video_path,
                output_path,
                source_id=source.source_id,
                detector_config=detector,
                tracker_config=tracker,
            )
    except (ContractError, RuntimeError, ValueError) as exc:
        logging.error("Track extraction failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

