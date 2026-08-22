from __future__ import annotations

from pathlib import Path

import pytest

from spittrack.common.errors import ContractError
from spittrack.data.contracts import METADATA_REQUIRED_COLUMNS, load_source_metadata


def valid_row(source_id: str = "S01") -> list[object]:
    values = {
        "source_id": source_id,
        "family_id": source_id,
        "file_name": f"{source_id}.mp4",
        "domain": "synthetic",
        "contains_spit": "true",
        "primary_label": "spitting",
        "negative_subtype": "",
        "generator_provider": "provider",
        "generator_model": "model",
        "generator_version": "v1",
        "generation_date": "2026-08-01",
        "prompt_id": "P01",
        "prompt_text": "controlled prompt",
        "seed": "42",
        "candidate_index": "1",
        "qc_status": "accepted",
        "qc_rejection_reason": "",
        "fps_original": "25",
        "fps_normalized": "25",
        "width": "1280",
        "height": "720",
        "duration_s": "10",
        "camera_height_m": "3.2",
        "camera_angle": "35",
        "lighting": "daylight",
        "occlusion_level": "none",
        "split_role": "",
        "sha256": "a" * 64,
    }
    return [values[column] for column in METADATA_REQUIRED_COLUMNS]


def test_header_only_contract_is_accepted_only_in_schema_mode(tmp_path: Path, write_csv) -> None:
    path = tmp_path / "metadata.csv"
    write_csv(path, list(METADATA_REQUIRED_COLUMNS), [])
    assert load_source_metadata(path, require_files=False, allow_empty=True) == []
    with pytest.raises(ContractError, match="no source rows"):
        load_source_metadata(path, require_files=False)


def test_missing_required_column_is_rejected(tmp_path: Path, write_csv) -> None:
    path = tmp_path / "metadata.csv"
    header = list(METADATA_REQUIRED_COLUMNS[:-1])
    write_csv(path, header, [valid_row()[:-1]])
    with pytest.raises(ContractError, match="sha256"):
        load_source_metadata(path, require_files=False)


def test_duplicate_source_id_is_rejected(tmp_path: Path, write_csv) -> None:
    path = tmp_path / "metadata.csv"
    write_csv(path, list(METADATA_REQUIRED_COLUMNS), [valid_row(), valid_row()])
    with pytest.raises(ContractError, match="duplicate source_id"):
        load_source_metadata(path, require_files=False)


def test_label_boolean_conflict_is_rejected(tmp_path: Path, write_csv) -> None:
    path = tmp_path / "metadata.csv"
    row = valid_row()
    row[list(METADATA_REQUIRED_COLUMNS).index("contains_spit")] = "false"
    with pytest.raises(ContractError, match="conflicts"):
        write_csv(path, list(METADATA_REQUIRED_COLUMNS), [row])
        load_source_metadata(path, require_files=False)


def test_real_source_cannot_have_train_role(tmp_path: Path, write_csv) -> None:
    path = tmp_path / "metadata.csv"
    row = valid_row("R01")
    index = {name: position for position, name in enumerate(METADATA_REQUIRED_COLUMNS)}
    row[index["domain"]] = "real_staged"
    row[index["family_id"]] = ""
    row[index["split_role"]] = "train"
    row[index["generator_provider"]] = ""
    row[index["generator_model"]] = ""
    row[index["generator_version"]] = ""
    with pytest.raises(ContractError, match="cannot have synthetic fold role"):
        write_csv(path, list(METADATA_REQUIRED_COLUMNS), [row])
        load_source_metadata(path, require_files=False)


def test_synthetic_source_requires_family_id(tmp_path: Path, write_csv) -> None:
    path = tmp_path / "metadata.csv"
    row = valid_row()
    row[list(METADATA_REQUIRED_COLUMNS).index("family_id")] = ""
    write_csv(path, list(METADATA_REQUIRED_COLUMNS), [row])
    with pytest.raises(ContractError, match="requires non-empty family_id"):
        load_source_metadata(path, require_files=False)


def test_real_source_requires_blank_family_id(tmp_path: Path, write_csv) -> None:
    path = tmp_path / "metadata.csv"
    row = valid_row("platform-positive")
    index = {name: position for position, name in enumerate(METADATA_REQUIRED_COLUMNS)}
    row[index["domain"]] = "real_staged"
    row[index["split_role"]] = "external_real"
    row[index["generator_provider"]] = ""
    row[index["generator_model"]] = ""
    row[index["generator_version"]] = ""
    write_csv(path, list(METADATA_REQUIRED_COLUMNS), [row])
    with pytest.raises(ContractError, match="must have blank family_id"):
        load_source_metadata(path, require_files=False)


def test_family_cannot_mix_positive_and_negative_sources(tmp_path: Path, write_csv) -> None:
    path = tmp_path / "metadata.csv"
    positive = valid_row("positive-variant")
    negative = valid_row("negative-variant")
    index = {name: position for position, name in enumerate(METADATA_REQUIRED_COLUMNS)}
    positive[index["family_id"]] = "shared-family"
    negative[index["family_id"]] = "shared-family"
    negative[index["contains_spit"]] = "false"
    negative[index["primary_label"]] = "non_spitting"
    negative[index["negative_subtype"]] = "normal_activity"
    write_csv(path, list(METADATA_REQUIRED_COLUMNS), [positive, negative])
    with pytest.raises(ContractError, match="mixes contains_spit=true and false"):
        load_source_metadata(path, require_files=False)
