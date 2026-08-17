"""Normalize validated raw source videos to 25 FPS without modifying originals."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spittrack.common.errors import ContractError
from spittrack.data.contracts import load_source_metadata
from spittrack.data.normalize import normalize_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/source_metadata.csv"))
    parser.add_argument("--synthetic-root", type=Path, default=Path("data/raw/synthetic"))
    parser.add_argument("--real-root", type=Path, default=Path("data/raw/real_staged"))
    parser.add_argument("--output-root", type=Path, default=Path("data/normalized"))
    parser.add_argument("--fps", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    roots = {"synthetic": args.synthetic_root, "real_staged": args.real_root}
    try:
        sources = load_source_metadata(args.metadata, roots)
        for source in sources:
            input_path = roots[source.domain] / source.file_name
            output_path = args.output_root / source.domain / f"{source.source_id}.mp4"
            if output_path.exists():
                raise ContractError(
                    f"Normalized output already exists; refusing implicit overwrite: {output_path}"
                )
            result = normalize_video(input_path, output_path, target_fps=args.fps)
            logging.info(
                "source=%s input_fps=%.6f output_fps=%.1f input_frames=%d output_frames=%d "
                "input_duration=%.6fs output_duration=%.6fs",
                source.source_id,
                result.input_fps,
                result.output_fps,
                result.input_frames,
                result.output_frames,
                result.input_duration_s,
                result.output_duration_s,
            )
    except (ContractError, RuntimeError) as exc:
        logging.error("Normalization failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

