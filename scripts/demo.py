"""Launch the local SpitTrack video demonstration."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spittrack.web.app import create_app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
