"""Run frozen YOLOv8s + ByteTrack over normalized staged-real holdout sources."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spittrack.common.config import load_yaml
from spittrack.common.errors import ContractError
from spittrack.data.contracts import SourceRecord, load_source_metadata
from spittrack.data.extract_tracks import extract_track_manifest


SYNTHETIC_TRACKING_ROOT = Path("data/tracking")


def _refuse_synthetic_tracking_tree(output_dir: Path) -> None:
    output = output_dir.resolve()
    synthetic = SYNTHETIC_TRACKING_ROOT.resolve()
    if output == synthetic or synthetic in output.parents:
        raise ContractError(
            "Real-holdout extraction refuses any output under the immutable synthetic "
            f"tracking tree: {output_dir}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/metadata/source_metadata.csv")
    )
    parser.add_argument(
        "--synthetic-root", type=Path, default=Path("data/raw/synthetic")
    )
    parser.add_argument(
        "--real-root", type=Path, default=Path("data/raw/real_staged")
    )
    parser.add_argument(
        "--normalized-root", type=Path, default=Path("data/normalized/real_staged")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/tracking_real/manifests")
    )
    parser.add_argument(
        "--detector-config", type=Path, default=Path("configs/detector.yaml")
    )
    parser.add_argument(
        "--tracker-config", type=Path, default=Path("configs/tracker.yaml")
    )
    parser.add_argument("--source-id", help="Process only one staged-real source")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        _refuse_synthetic_tracking_tree(args.output_dir)
        sources = load_source_metadata(
            args.metadata,
            {"synthetic": args.synthetic_root, "real_staged": args.real_root},
        )
        real_sources = [source for source in sources if source.domain == "real_staged"]
        if args.source_id:
            real_sources = [
                source for source in real_sources if source.source_id == args.source_id
            ]
            if not real_sources:
                raise ContractError(f"Unknown real_staged --source-id {args.source_id!r}")
        if not real_sources:
            raise ContractError("No validated real_staged sources are available")

        jobs: list[tuple[SourceRecord, Path, Path]] = []
        for source in real_sources:
            video_path = args.normalized_root / f"{source.source_id}.mp4"
            output_path = args.output_dir / f"{source.source_id}.csv"
            if not video_path.is_file():
                raise ContractError(f"Normalized staged-real source does not exist: {video_path}")
            if output_path.exists():
                raise ContractError(
                    "Real tracking manifest already exists; immutable artifacts are not "
                    f"overwritten: {output_path}"
                )
            jobs.append((source, video_path, output_path))

        # Load and pass through the exact frozen dictionaries without changing parameters.
        detector = load_yaml(args.detector_config)
        tracker = load_yaml(args.tracker_config)
        run_ids: set[str] = set()
        for source, video_path, output_path in jobs:
            run_ids.add(
                extract_track_manifest(
                    video_path,
                    output_path,
                    source_id=source.source_id,
                    detector_config=detector,
                    tracker_config=tracker,
                )
            )
        if len(run_ids) != 1:
            raise ContractError(
                f"Real sources unexpectedly produced multiple tracking_run_id values: {run_ids}"
            )
        run_id = next(iter(run_ids))
    except (ContractError, RuntimeError, ValueError, OSError) as exc:
        logging.error("Real-holdout track extraction failed: %s", exc)
        return 2

    logging.info(
        "Completed staged-real tracking: sources=%d manifests=%s",
        len(real_sources),
        args.output_dir,
    )
    print(f"real_tracking_run_id={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
