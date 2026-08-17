"""Run frozen final-model inference over one arbitrary video (CPU by default)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spittrack.common.errors import ContractError
from spittrack.inference.pipeline import run_video_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-id", help="Stable event source ID; defaults to the input filename stem")
    parser.add_argument("--final-model-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--detector-config", type=Path, default=Path("configs/detector.yaml"))
    parser.add_argument("--tracker-config", type=Path, default=Path("configs/tracker.yaml"))
    parser.add_argument("--inference-config", type=Path, default=Path("configs/inference.yaml"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        result = run_video_inference(args.input, args.final_model_provenance, args.output_dir, source_id=args.source_id, device_name=args.device, data_config_path=args.data_config, detector_config_path=args.detector_config, tracker_config_path=args.tracker_config, inference_config_path=args.inference_config, batch_size=args.batch_size)
        logging.info("Inference complete: %s", json.dumps(result, sort_keys=True))
    except (ContractError, FileNotFoundError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        logging.error("Inference failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
