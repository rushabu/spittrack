from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from spittrack.data.contracts import SourceRecord


@pytest.fixture
def source_factory() -> Callable[..., SourceRecord]:
    def make(
        source_id: str,
        contains_spit: bool,
        *,
        domain: str = "synthetic",
        duration_s: float = 10.0,
        family_id: str | None = None,
    ) -> SourceRecord:
        return SourceRecord(
            source_id=source_id,
            family_id=(family_id or source_id) if domain == "synthetic" else None,
            file_name=f"{source_id}.mp4",
            domain=domain,
            contains_spit=contains_spit,
            primary_label="spitting" if contains_spit else "non_spitting",
            negative_subtype=None if contains_spit else "normal_activity",
            duration_s=duration_s,
            fps_original=25.0,
            fps_normalized=25.0,
            width=1280,
            height=720,
            split_role="external_real" if domain == "real_staged" else "",
            sha256="0" * 64,
            row={},
        )

    return make


@pytest.fixture
def write_csv() -> Callable[[Path, list[str], list[list[object]]], None]:
    def write(path: Path, header: list[str], rows: list[list[object]]) -> None:
        import csv

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)

    return write
