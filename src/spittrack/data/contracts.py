"""Strict CSV contracts for sources and track-level action annotations."""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from spittrack.common.errors import ContractError

METADATA_REQUIRED_COLUMNS = (
    "source_id",
    "family_id",
    "file_name",
    "domain",
    "contains_spit",
    "primary_label",
    "negative_subtype",
    "generator_provider",
    "generator_model",
    "generator_version",
    "generation_date",
    "prompt_id",
    "prompt_text",
    "seed",
    "candidate_index",
    "qc_status",
    "qc_rejection_reason",
    "fps_original",
    "fps_normalized",
    "width",
    "height",
    "duration_s",
    "camera_height_m",
    "camera_angle",
    "lighting",
    "occlusion_level",
    "split_role",
    "sha256",
)

METADATA_OPTIONAL_COLUMNS = (
    "location_type",
    "permission_reference",
    "consent_reference",
    "main_actor_count",
    "crowd_count",
)

ANNOTATION_REQUIRED_COLUMNS = (
    "source_id",
    "tracking_run_id",
    "track_id",
    "actor_role",
    "action_label",
    "action_start_s",
    "action_end_s",
    "negative_subtype",
    "annotation_confidence",
    "notes",
)

DOMAINS = frozenset({"synthetic", "real_staged"})
PRIMARY_LABELS = frozenset({"spitting", "non_spitting"})
NEGATIVE_SUBTYPES = frozenset(
    {
        "drink",
        "cough",
        "sneeze",
        "eat",
        "wipe_mouth",
        "face_touch",
        "bend_pickup",
        "mask_adjust",
        "talk_laugh",
        "normal_activity",
    }
)
ACTION_LABELS = frozenset({"spitting", *NEGATIVE_SUBTYPES})
SYNTHETIC_FOLD_ROLES = frozenset({"train", "validation", "val", "test"})
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class SourceRecord:
    """Validated metadata for one source video."""

    source_id: str
    family_id: str | None
    file_name: str
    domain: str
    contains_spit: bool
    primary_label: str
    negative_subtype: str | None
    duration_s: float
    fps_original: float
    fps_normalized: float
    width: int
    height: int
    split_role: str
    sha256: str
    row: Mapping[str, str]


@dataclass(frozen=True)
class ActionInterval:
    """Validated action interval attached to a specific tracker trajectory."""

    source_id: str
    tracking_run_id: str
    track_id: int
    actor_role: str
    action_label: str
    start_s: float
    end_s: float
    negative_subtype: str | None
    annotation_confidence: float
    notes: str

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def _read_csv(path: Path, required_columns: Iterable[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ContractError(f"Required CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ContractError(f"CSV has no header: {path}")
        duplicate_headers = sorted(
            {name for name in reader.fieldnames if reader.fieldnames.count(name) > 1}
        )
        if duplicate_headers:
            raise ContractError(f"CSV contains duplicate columns {duplicate_headers}: {path}")
        missing = sorted(set(required_columns) - set(reader.fieldnames))
        if missing:
            raise ContractError(f"CSV is missing required columns {missing}: {path}")
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def _parse_bool(value: str, field: str, context: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ContractError(f"{context}: {field} must be true or false, got {value!r}")


def _parse_float(value: str, field: str, context: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ContractError(f"{context}: {field} must be numeric, got {value!r}") from exc


def _parse_int(value: str, field: str, context: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ContractError(f"{context}: {field} must be an integer, got {value!r}") from exc
    return parsed


def _resolve_source_path(record: SourceRecord, raw_roots: Mapping[str, Path]) -> Path:
    if record.domain not in raw_roots:
        raise ContractError(f"No raw-video root configured for domain {record.domain!r}")
    relative = Path(record.file_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ContractError(
            f"source {record.source_id}: file_name must be a safe relative path, got {record.file_name!r}"
        )
    root = raw_roots[record.domain].resolve()
    candidate = (root / relative).resolve()
    if root != candidate and root not in candidate.parents:
        raise ContractError(f"source {record.source_id}: file path escapes its domain root")
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_metadata(
    path: Path,
    raw_roots: Mapping[str, Path] | None = None,
    *,
    require_files: bool = True,
    allow_empty: bool = False,
) -> list[SourceRecord]:
    """Load and validate source metadata, optionally checking referenced MP4 files."""
    rows = _read_csv(path, METADATA_REQUIRED_COLUMNS)
    if not rows and not allow_empty:
        raise ContractError(
            f"Metadata contains no source rows: {path}. Add at least one source before validation."
        )

    records: list[SourceRecord] = []
    seen_ids: set[str] = set()
    family_classes: dict[str, bool] = {}
    for row_number, row in enumerate(rows, start=2):
        context = f"metadata row {row_number}"
        source_id = row["source_id"]
        if not source_id:
            raise ContractError(f"{context}: source_id cannot be blank")
        if source_id in seen_ids:
            raise ContractError(f"{context}: duplicate source_id {source_id!r}")
        seen_ids.add(source_id)

        domain = row["domain"]
        if domain not in DOMAINS:
            raise ContractError(f"{context}: domain must be one of {sorted(DOMAINS)}, got {domain!r}")
        contains_spit = _parse_bool(row["contains_spit"], "contains_spit", context)
        family_id = row["family_id"] or None
        if domain == "synthetic":
            if family_id is None:
                raise ContractError(f"{context}: synthetic source requires non-empty family_id")
            prior_class = family_classes.get(family_id)
            if prior_class is not None and prior_class != contains_spit:
                raise ContractError(
                    f"{context}: family_id {family_id!r} mixes contains_spit=true and false sources"
                )
            family_classes[family_id] = contains_spit
        elif family_id is not None:
            raise ContractError(f"{context}: real_staged source must have blank family_id")
        primary_label = row["primary_label"]
        if primary_label not in PRIMARY_LABELS:
            raise ContractError(
                f"{context}: primary_label must be one of {sorted(PRIMARY_LABELS)}, got {primary_label!r}"
            )
        expected_label = "spitting" if contains_spit else "non_spitting"
        if primary_label != expected_label:
            raise ContractError(
                f"{context}: contains_spit={contains_spit} conflicts with primary_label={primary_label!r}"
            )
        negative_subtype = row["negative_subtype"] or None
        if negative_subtype is not None and negative_subtype not in NEGATIVE_SUBTYPES:
            raise ContractError(
                f"{context}: unknown negative_subtype {negative_subtype!r}; "
                f"expected one of {sorted(NEGATIVE_SUBTYPES)}"
            )
        if contains_spit and negative_subtype is not None:
            raise ContractError(f"{context}: positive source cannot have a source-level negative_subtype")
        if not contains_spit and negative_subtype is None:
            raise ContractError(f"{context}: non-spitting source requires negative_subtype")

        duration_s = _parse_float(row["duration_s"], "duration_s", context)
        fps_original = _parse_float(row["fps_original"], "fps_original", context)
        fps_normalized = _parse_float(row["fps_normalized"], "fps_normalized", context)
        width = _parse_int(row["width"], "width", context)
        height = _parse_int(row["height"], "height", context)
        if duration_s <= 0 or fps_original <= 0 or fps_normalized <= 0 or width <= 0 or height <= 0:
            raise ContractError(f"{context}: duration, FPS, width, and height must all be positive")
        if abs(fps_normalized - 25.0) > 0.01:
            raise ContractError(f"{context}: fps_normalized must be 25, got {fps_normalized}")

        split_role = row["split_role"].lower()
        if domain == "real_staged" and split_role in SYNTHETIC_FOLD_ROLES:
            raise ContractError(
                f"{context}: real_staged source {source_id} cannot have synthetic fold role {split_role!r}"
            )
        if not row["file_name"]:
            raise ContractError(f"{context}: file_name cannot be blank")
        sha256 = row["sha256"]
        if not _SHA256_RE.fullmatch(sha256):
            raise ContractError(f"{context}: sha256 must contain exactly 64 hexadecimal characters")

        if domain == "synthetic":
            provenance = ("generator_provider", "generator_model", "generator_version", "prompt_text")
            missing_provenance = [field for field in provenance if not row[field]]
            if missing_provenance:
                raise ContractError(f"{context}: synthetic source is missing provenance {missing_provenance}")
        if row["qc_status"].lower() != "accepted":
            raise ContractError(
                f"{context}: source_metadata contains retained sources only; qc_status must be 'accepted'"
            )

        for optional_count in ("main_actor_count", "crowd_count"):
            if optional_count in row and row[optional_count]:
                if _parse_int(row[optional_count], optional_count, context) < 0:
                    raise ContractError(f"{context}: {optional_count} cannot be negative")

        record = SourceRecord(
            source_id=source_id,
            family_id=family_id,
            file_name=row["file_name"],
            domain=domain,
            contains_spit=contains_spit,
            primary_label=primary_label,
            negative_subtype=negative_subtype,
            duration_s=duration_s,
            fps_original=fps_original,
            fps_normalized=fps_normalized,
            width=width,
            height=height,
            split_role=split_role,
            sha256=sha256.lower(),
            row=row,
        )
        if require_files:
            if raw_roots is None:
                raise ContractError("raw_roots must be provided when require_files=True")
            source_path = _resolve_source_path(record, raw_roots)
            if not source_path.is_file():
                raise ContractError(
                    f"source {source_id}: referenced video does not exist in {domain}: {source_path}"
                )
            actual_hash = _sha256_file(source_path)
            if actual_hash != record.sha256:
                raise ContractError(
                    f"source {source_id}: sha256 mismatch for {source_path}; "
                    f"metadata={record.sha256}, actual={actual_hash}"
                )
        records.append(record)
    return records


def load_action_annotations(
    path: Path,
    sources: Iterable[SourceRecord],
    *,
    known_track_pairs: set[tuple[str, int]] | None = None,
    allow_empty: bool = False,
) -> list[ActionInterval]:
    """Load action intervals and validate source, timing, label, and optional track references."""
    rows = _read_csv(path, ANNOTATION_REQUIRED_COLUMNS)
    if not rows and not allow_empty:
        raise ContractError(
            f"Annotation CSV contains no intervals: {path}. Add annotations before building windows."
        )
    source_map = {source.source_id: source for source in sources}
    intervals: list[ActionInterval] = []
    for row_number, row in enumerate(rows, start=2):
        context = f"annotation row {row_number}"
        source_id = row["source_id"]
        if source_id not in source_map:
            raise ContractError(f"{context}: unknown source_id {source_id!r}")
        tracking_run_id = row["tracking_run_id"]
        if not tracking_run_id:
            raise ContractError(f"{context}: tracking_run_id cannot be blank")
        track_id = _parse_int(row["track_id"], "track_id", context)
        if track_id < 0:
            raise ContractError(f"{context}: track_id cannot be negative")
        if known_track_pairs is not None and (source_id, track_id) not in known_track_pairs:
            raise ContractError(
                f"{context}: source/track pair ({source_id}, {track_id}) is absent from tracking manifests"
            )
        action_label = row["action_label"]
        if action_label not in ACTION_LABELS:
            raise ContractError(
                f"{context}: action_label must be one of {sorted(ACTION_LABELS)}, got {action_label!r}"
            )
        start_s = _parse_float(row["action_start_s"], "action_start_s", context)
        end_s = _parse_float(row["action_end_s"], "action_end_s", context)
        if start_s < 0 or end_s <= start_s:
            raise ContractError(f"{context}: interval must satisfy 0 <= start_s < end_s")
        if end_s > source_map[source_id].duration_s + 1e-6:
            raise ContractError(
                f"{context}: interval ends at {end_s}s beyond source duration "
                f"{source_map[source_id].duration_s}s"
            )
        confidence = _parse_float(row["annotation_confidence"], "annotation_confidence", context)
        if not 0.0 <= confidence <= 1.0:
            raise ContractError(f"{context}: annotation_confidence must be in [0, 1]")
        negative_subtype = row["negative_subtype"] or None
        if negative_subtype is not None and negative_subtype not in NEGATIVE_SUBTYPES:
            raise ContractError(f"{context}: unknown negative_subtype {negative_subtype!r}")
        if action_label == "spitting" and negative_subtype is not None:
            raise ContractError(f"{context}: spitting interval cannot have negative_subtype")
        if action_label != "spitting" and negative_subtype not in {None, action_label}:
            raise ContractError(
                f"{context}: negative_subtype {negative_subtype!r} conflicts with action_label {action_label!r}"
            )
        intervals.append(
            ActionInterval(
                source_id=source_id,
                tracking_run_id=tracking_run_id,
                track_id=track_id,
                actor_role=row["actor_role"],
                action_label=action_label,
                start_s=start_s,
                end_s=end_s,
                negative_subtype=negative_subtype,
                annotation_confidence=confidence,
                notes=row["notes"],
            )
        )
    return intervals
