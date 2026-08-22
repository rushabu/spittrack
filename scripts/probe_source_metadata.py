"""Populate file-derived source metadata by probing every referenced MP4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
import sys

import cv2


PROBED_FIELDS = ("fps_original", "width", "height", "duration_s", "sha256")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/metadata/source_metadata.csv"),
    )
    parser.add_argument(
        "--synthetic-root",
        type=Path,
        default=Path("data/raw/synthetic"),
    )
    parser.add_argument(
        "--real-root",
        type=Path,
        default=Path("data/raw/real_staged"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write to another CSV instead of updating --metadata in place.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_number(value: float, places: int = 3) -> str:
    return f"{value:.{places}f}".rstrip("0").rstrip(".")


def _resolve_video(root: Path, file_name: str, source_id: str) -> Path:
    relative = Path(file_name)
    if not file_name or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            f"source {source_id}: file_name must be a safe relative path, got {file_name!r}"
        )
    resolved_root = root.resolve()
    candidate = (resolved_root / relative).resolve()
    if resolved_root != candidate and resolved_root not in candidate.parents:
        raise ValueError(f"source {source_id}: file_name escapes its domain root")
    if not candidate.is_file():
        raise FileNotFoundError(f"source {source_id}: MP4 does not exist: {candidate}")
    return candidate


def _probe_video(path: Path, source_id: str) -> dict[str, str]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"source {source_id}: OpenCV could not open {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"source {source_id}: invalid FPS {fps!r} reported by OpenCV")
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError(
            f"source {source_id}: invalid video geometry/frame count "
            f"({width}x{height}, {frame_count} frames)"
        )
    return {
        "fps_original": _format_number(fps),
        "width": str(width),
        "height": str(height),
        "duration_s": _format_number(frame_count / fps),
        "sha256": _sha256(path),
    }


def main() -> int:
    args = parse_args()
    if not args.metadata.is_file():
        print(f"ERROR: metadata CSV does not exist: {args.metadata}", file=sys.stderr)
        return 2

    roots = {
        "synthetic": args.synthetic_root,
        "real_staged": args.real_root,
    }
    try:
        with args.metadata.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise ValueError("metadata CSV has no header")
            required = {"source_id", "file_name", "domain", *PROBED_FIELDS}
            missing = sorted(required - set(fieldnames))
            if missing:
                raise ValueError(f"metadata CSV is missing columns {missing}")
            rows = list(reader)
        if not rows:
            raise ValueError("metadata CSV contains no source rows")

        seen_ids: set[str] = set()
        for row in rows:
            source_id = (row.get("source_id") or "").strip()
            if not source_id or source_id in seen_ids:
                raise ValueError(f"source_id is blank or duplicated: {source_id!r}")
            seen_ids.add(source_id)
            domain = (row.get("domain") or "").strip()
            if domain not in roots:
                raise ValueError(f"source {source_id}: unsupported domain {domain!r}")
            video_path = _resolve_video(
                roots[domain], (row.get("file_name") or "").strip(), source_id
            )
            values = _probe_video(video_path, source_id)
            row.update(values)
            print(
                f"{source_id}: fps={values['fps_original']} "
                f"size={values['width']}x{values['height']} "
                f"duration={values['duration_s']}s sha256={values['sha256']}"
            )

        destination = args.output or args.metadata
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, ValueError) as exc:
        print(f"ERROR: metadata probing failed: {exc}", file=sys.stderr)
        return 2

    print(f"Updated {len(rows)} rows in {destination}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
