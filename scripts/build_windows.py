"""Build a deterministic fold-independent temporal-window index from tracker manifests."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spittrack.common.errors import ContractError, WindowError
from spittrack.data.contracts import load_action_annotations, load_source_metadata
from spittrack.data.tracking import load_tracking_manifests
from spittrack.data.materialize import materialize_window_clips
from spittrack.data.windows import build_window_index, write_window_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/source_metadata.csv"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/action_intervals.csv"))
    parser.add_argument("--synthetic-root", type=Path, default=Path("data/raw/synthetic"))
    parser.add_argument("--real-root", type=Path, default=Path("data/raw/real_staged"))
    parser.add_argument("--tracking-dir", type=Path, default=Path("data/tracking/manifests"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/windows/windows.csv"))
    parser.add_argument("--normalized-root", type=Path, default=Path("data/normalized/synthetic"))
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Write metadata index without decoding and materializing crop arrays.",
    )
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
        track_frames = load_tracking_manifests(args.tracking_dir, synthetic)
        known_pairs = {(frame.source_id, frame.track_id) for frame in track_frames}
        annotations = load_action_annotations(args.annotations, sources)
        synthetic_ids = {source.source_id for source in synthetic}
        synthetic_annotations = [item for item in annotations if item.source_id in synthetic_ids]
        missing_pairs = sorted(
            {
                (item.source_id, item.track_id)
                for item in synthetic_annotations
                if (item.source_id, item.track_id) not in known_pairs
            }
        )
        if missing_pairs:
            raise ContractError(
                f"Synthetic annotation source/track pairs are absent from tracking manifests: {missing_pairs}"
            )
        windows, summary = build_window_index(synthetic, track_frames, synthetic_annotations)
        if not args.index_only:
            materialize_window_clips(
                windows,
                track_frames,
                normalized_root=args.normalized_root,
                clips_root=args.output.parent / "clips",
            )
        write_window_index(windows, args.output)
    except (ContractError, WindowError) as exc:
        logging.error("Window construction failed: %s", exc)
        return 2
    logging.info(
        "Window index written to %s: sources=%d tracks=%d windows=%d ignored=%d "
        "rejected_incomplete=%d",
        args.output,
        summary.source_count,
        summary.track_count,
        summary.emitted_windows,
        summary.ignored_windows,
        summary.rejected_incomplete_windows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
