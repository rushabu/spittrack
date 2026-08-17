"""Consistent console and durable run-file logging."""

from __future__ import annotations

import logging
from pathlib import Path


def attach_run_log(path: Path) -> None:
    """Attach exactly one active UTF-8 run log to the root logger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_spittrack_run_file", False):
            root.removeHandler(handler)
            handler.close()
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    handler._spittrack_run_file = True  # type: ignore[attr-defined]
    root.addHandler(handler)
