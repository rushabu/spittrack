from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_real_extraction_cli_refuses_synthetic_tracking_tree() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "extract_tracks_real.py"),
            "--output-dir",
            str(root / "data" / "tracking" / "manifests"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "immutable synthetic tracking tree" in result.stderr
