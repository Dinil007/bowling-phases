"""
Manifest ingestion and integrity verification (Brief 4A).

Every per-clip failure raises a PipelineError subclass carrying an explicit exit code,
so the entry point can report it clearly and exit non-zero.
"""
import csv
import hashlib
import os
from dataclasses import dataclass
from typing import List, Optional

REQUIRED_COLUMNS = ("clip_id", "file_name", "sha256")


class PipelineError(Exception):
    code = 20
    tag = "PIPELINE_ERROR"


class ManifestError(PipelineError):
    code = 1
    tag = "MANIFEST_INVALID"


class MissingFileError(PipelineError):
    code = 10
    tag = "MISSING_FILE"


class UnreadableFileError(PipelineError):
    code = 11
    tag = "UNREADABLE_FILE"


class ChecksumMismatchError(PipelineError):
    code = 12
    tag = "CHECKSUM_MISMATCH"


class NoPersonDetectedError(PipelineError):
    code = 13
    tag = "NO_PERSON"


class AmbiguousBowlerError(PipelineError):
    code = 14
    tag = "AMBIGUOUS_BOWLER"


@dataclass
class ManifestEntry:
    clip_id: str
    file_name: str          # as written in the manifest
    path: Optional[str]     # resolved absolute path (None if not found)
    sha256: str
    frame_rate: Optional[float]
    resolution: str
    duration: Optional[float]
    camera_view: str
    athlete_id: str


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _float_or_none(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def resolve_path(file_name: str, manifest_dir: str) -> Optional[str]:
    """Resolve a manifest file path: absolute, relative to the manifest, then relative to CWD."""
    candidates = [file_name] if os.path.isabs(file_name) else [
        os.path.join(manifest_dir, file_name),
        os.path.join(os.getcwd(), file_name),
        os.path.join(manifest_dir, os.path.basename(file_name)),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


def read_manifest(manifest_path: str) -> List[ManifestEntry]:
    """Parse the manifest only. Raises ManifestError if the manifest itself is unusable."""
    if not os.path.isfile(manifest_path):
        raise ManifestError(f"Manifest not found: {manifest_path}")
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = [c.strip() for c in (reader.fieldnames or [])]
        missing = [c for c in REQUIRED_COLUMNS if c not in cols]
        if missing:
            raise ManifestError(f"Manifest {manifest_path} is missing columns: {missing}")
        entries = []
        for row in reader:
            row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            if not row.get("clip_id"):
                continue
            entries.append(ManifestEntry(
                clip_id=row["clip_id"],
                file_name=row["file_name"],
                path=resolve_path(row["file_name"], manifest_dir),
                sha256=row["sha256"].lower(),
                frame_rate=_float_or_none(row.get("frame_rate")),
                resolution=row.get("resolution", ""),
                duration=_float_or_none(row.get("duration")),
                camera_view=row.get("camera_view", "unknown") or "unknown",
                athlete_id=row.get("athlete_id", "") or f"ATH-PLACEHOLDER-{row['clip_id']}",
            ))
    if not entries:
        raise ManifestError(f"Manifest {manifest_path} contains no clips")
    return entries


def verify_entry(entry: ManifestEntry, verify_checksum: bool = True) -> str:
    """Existence + checksum check for one clip. Returns the SHA-256 actually computed."""
    if entry.path is None:
        raise MissingFileError(f"[{entry.clip_id}] video file not found: {entry.file_name}")
    try:
        actual = compute_sha256(entry.path)
    except OSError as e:
        raise UnreadableFileError(f"[{entry.clip_id}] cannot read {entry.path}: {e}")
    if verify_checksum and actual != entry.sha256:
        raise ChecksumMismatchError(
            f"[{entry.clip_id}] checksum mismatch for {entry.path}: manifest {entry.sha256}, actual {actual}")
    return actual


def load_manifest(manifest_path: str, verify_checksums: bool = True) -> List[ManifestEntry]:
    """Backwards-compatible helper: parse and verify every entry, raising on the first failure."""
    entries = read_manifest(manifest_path)
    for e in entries:
        verify_entry(e, verify_checksums)
    return entries
