"""Guarded, durable, one-time staged-real stress test."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
from datetime import datetime, timezone

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spittrack.common.errors import ContractError
from spittrack.data.contracts import _sha256_file, load_action_annotations, load_source_metadata
from spittrack.inference.pipeline import run_video_inference

ACK_FLAG = "--i-understand-this-is-one-time"
RECEIPT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "staged_real_execution_receipt.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-model-provenance", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/source_metadata.csv"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/action_intervals.csv"))
    parser.add_argument("--real-root", type=Path, default=Path("data/raw/real_staged"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/staged_real"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--i-understand-this-is-one-time", action="store_true")
    return parser.parse_args()


def _provenance(path: Path) -> dict:
    if not path.is_file():
        raise ContractError(f"Final-model provenance is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"model_name", "checkpoint_path", "median_cv_best_epoch", "median_fold_validation_threshold", "cv_summary_sha256", "config_sha256", "final_training_uses_validation_set"}
    missing = sorted(required - set(payload))
    if missing:
        raise ContractError(f"Final-model provenance is incomplete; missing fields: {missing}")
    if payload["final_training_uses_validation_set"] is not False:
        raise ContractError("Final provenance does not certify no-validation final training")
    checkpoint = Path(str(payload["checkpoint_path"]))
    if not checkpoint.is_absolute():
        checkpoint = path.parent / checkpoint
    snapshot = Path(str(payload.get("config_snapshot_path", "config_snapshot.json")))
    if not snapshot.is_absolute():
        snapshot = path.parent / snapshot
    if not checkpoint.is_file() or not snapshot.is_file():
        raise ContractError(
            f"Final-model provenance references missing checkpoint/config files: {checkpoint}, {snapshot}"
        )
    return payload


def _iou(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    overlap = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    union = max(left_end, right_end) - min(left_start, right_start)
    return overlap / union if union else 0.0


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.i_understand_this_is_one_time:
        logging.error("Refusing staged-real evaluation without explicit %s acknowledgement", ACK_FLAG)
        return 2
    try:
        provenance = _provenance(args.final_model_provenance)
        logging.info("Frozen staged-real provenance: model=%s checkpoint=%s epoch=%s threshold=%s cv_summary_sha256=%s config_sha256=%s", provenance["model_name"], provenance["checkpoint_path"], provenance["median_cv_best_epoch"], provenance["median_fold_validation_threshold"], provenance["cv_summary_sha256"], provenance["config_sha256"])
        receipt_path = RECEIPT_PATH
        if receipt_path.exists():
            raise ContractError(f"One-time staged-real test was already started; receipt exists: {receipt_path}")
        sources = load_source_metadata(args.metadata, require_files=False)
        real_sources = [source for source in sources if source.domain == "real_staged"]
        if not real_sources:
            raise ContractError("No validated real_staged sources are available")
        annotations = load_action_annotations(args.annotations, sources, allow_empty=True)
        paths: dict[str, Path] = {}
        for source in real_sources:
            relative = Path(source.file_name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ContractError(f"Unsafe staged-real file_name: {source.file_name}")
            video = (args.real_root / relative).resolve()
            root = args.real_root.resolve()
            if root != video and root not in video.parents:
                raise ContractError("Staged-real path escapes configured root")
            if not video.is_file():
                raise ContractError(f"Staged-real clip is missing: {video}")
            if _sha256_file(video) != source.sha256:
                raise ContractError(f"Staged-real SHA-256 mismatch: {video}")
            paths[source.source_id] = video
        args.output_dir.mkdir(parents=True, exist_ok=True)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt = {"status": "started", "started_at_utc": datetime.now(timezone.utc).isoformat(), "model_name": provenance["model_name"], "checkpoint_path": provenance["checkpoint_path"], "threshold": provenance["median_fold_validation_threshold"], "source_ids": [source.source_id for source in real_sources]}
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows = []
        for source in real_sources:
            clip_dir = args.output_dir / "clips" / source.source_id
            result = run_video_inference(paths[source.source_id], args.final_model_provenance, clip_dir, device_name=args.device)
            events = json.loads((clip_dir / "events.json").read_text(encoding="utf-8"))
            truth = [item for item in annotations if item.source_id == source.source_id and item.action_label == "spitting"]
            run_ids = {item.tracking_run_id for item in truth}
            if run_ids and run_ids != {result["tracking_run_id"]}:
                raise ContractError(f"{source.source_id}: annotation/inference tracking_run_id mismatch")
            matches = [(event, item, _iou(float(event["start_s"]), float(event["end_s"]), item.start_s, item.end_s)) for event in events for item in truth]
            best = max(matches, key=lambda value: value[2]) if matches else None
            matched_ids = {event["event_id"] for event, item, overlap in matches if overlap >= 0.5 and int(event["track_id"]) == item.track_id}
            rows.append({"source_id": source.source_id, "detected": bool(events), "correct_track_detection": bool(best and best[2] >= 0.5 and int(best[0]["track_id"]) == best[1].track_id), "predicted_track_id": best[0]["track_id"] if best else "", "predicted_start_s": best[0]["start_s"] if best else "", "predicted_end_s": best[0]["end_s"] if best else "", "confidence": best[0]["confidence"] if best else "", "false_alert_count": len(events) - len(matched_ids), "notes": "descriptive one-time stress test; no aggregate accuracy claimed"})
        report = args.output_dir / "staged_real_stress_test.csv"
        with report.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        receipt["status"] = "complete"
        receipt["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        receipt["report"] = str(report.resolve())
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (ContractError, FileNotFoundError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        logging.error("Staged-real test refused or failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
