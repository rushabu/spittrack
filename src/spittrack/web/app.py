"""A local Flask interface over the existing isolated inference command."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
from threading import Lock, Thread
from uuid import uuid4

from flask import Flask, abort, jsonify, render_template, request, send_file


ROOT = Path(__file__).resolve().parents[3]


def create_app(*, project_root: Path = ROOT, runner=None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024
    runs = project_root / "outputs" / "web_demo"
    provenance = project_root / "outputs" / "final_model" / "final_model_provenance.json"
    busy = Lock()
    status_lock = Lock()

    def job_dir(job_id):
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            abort(404)
        directory = runs / job_id
        if not directory.is_dir():
            abort(404)
        return directory

    def save_status(directory, **values):
        with status_lock:
            (directory / "status.json").write_text(json.dumps(values), encoding="utf-8")

    def run_job(directory, input_path):
        try:
            save_status(directory, state="running", message="Normalizing video, tracking people, and classifying windows…")
            result_dir = directory / "result"
            if runner is not None:
                runner(input_path, provenance, result_dir)
            else:
                with (directory / "process.log").open("w", encoding="utf-8") as log:
                    subprocess.run(
                        [sys.executable, str(project_root / "scripts" / "run_inference.py"),
                         "--input", str(input_path), "--source-id", "demo",
                         "--final-model-provenance", str(provenance),
                         "--output-dir", str(result_dir)],
                        cwd=project_root, stdout=log, stderr=subprocess.STDOUT, check=True,
                    )
            save_status(directory, state="running", message="Preparing annotated video and evidence for browser playback…")
            import imageio_ffmpeg

            videos = [result_dir / "annotated.mp4", *sorted((result_dir / "evidence").glob("*.mp4"))]
            for video in videos:
                subprocess.run(
                    [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(video),
                     "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                     "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-movflags", "+faststart",
                     str(video.with_name(video.stem + "_browser.mp4"))],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True,
                )
            save_status(directory, state="complete", message="Analysis complete")
        except Exception:
            app.logger.exception("Demo analysis failed")
            save_status(directory, state="failed", message="Analysis failed. Open the run log below for details. Check that the video contains a visible person for at least two seconds and the model dependencies are installed.")
        finally:
            busy.release()

    @app.get("/")
    def index():
        model = {}
        error = None
        try:
            model = json.loads(provenance.read_text(encoding="utf-8"))
            checkpoint = Path(model["checkpoint_path"])
            if not checkpoint.is_absolute():
                checkpoint = provenance.parent / checkpoint
            if not checkpoint.is_file():
                error = "Final checkpoint is missing. Restore outputs/final_model/final.pt before running."
        except (OSError, ValueError, KeyError):
            error = "Final model provenance is unavailable. Complete final training or restore outputs/final_model."
        return render_template("index.html", model=model, error=error)

    @app.post("/api/runs")
    def start():
        video = request.files.get("video")
        if video is None or not video.filename or Path(video.filename).suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            return jsonify(error="Choose an MP4, MOV, AVI, MKV, or WebM video."), 400
        if not provenance.is_file():
            return jsonify(error="Final-model provenance is missing."), 400
        if not busy.acquire(blocking=False):
            return jsonify(error="An analysis is already running. Wait for it to finish."), 409
        try:
            job_id = uuid4().hex
            directory = runs / job_id
            directory.mkdir(parents=True)
            input_path = directory / ("input" + Path(video.filename).suffix.lower())
            video.save(input_path)
            if not input_path.stat().st_size:
                busy.release()
                return jsonify(error="The uploaded video is empty."), 400
            save_status(directory, state="queued", message="Starting analysis…")
            Thread(target=run_job, args=(directory, input_path), daemon=True).start()
        except Exception:
            busy.release()
            raise
        return jsonify(id=job_id), 202

    @app.get("/api/runs/<job_id>")
    def status(job_id):
        directory = job_dir(job_id)
        with status_lock:
            result = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        log = directory / "process.log"
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 6000))
                result["log"] = stream.read().decode("utf-8", errors="replace")
        if result["state"] == "complete":
            result["summary"] = json.loads((directory / "result" / "inference_summary.json").read_text(encoding="utf-8"))
            result["events"] = json.loads((directory / "result" / "events.json").read_text(encoding="utf-8"))
        return jsonify(result)

    @app.get("/runs/<job_id>/files/<path:name>")
    def artifact(job_id, name):
        directory = job_dir(job_id)
        allowed = {"annotated_browser.mp4", "annotated.mp4", "events.csv", "events.json", "inference_summary.json"}
        if name == "process.log":
            target = directory / name
        elif name in allowed or re.fullmatch(r"evidence/event_\d{4,}(?:_browser)?\.mp4", name):
            target = directory / "result" / name
        else:
            abort(404)
        if not target.is_file():
            abort(404)
        return send_file(target, conditional=True, as_attachment=request.args.get("download") == "1")

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify(error="Video exceeds the 512 MB upload limit."), 413

    return app
