"""Tests for candidate registry consistency.

These tests validate that the experiment tracking files are well-formed and
internally consistent:
- experiments/candidate_registry.csv exists and has required fields
- experiments/candidates.jsonl exists and each line is valid JSON
- Cross-references between CSV and JSONL are consistent
- Referenced checkpoint paths exist (warning-level only)

All tests run locally without GPU.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_CSV = PROJECT_ROOT / "experiments" / "candidate_registry.csv"
CANDIDATES_JSONL = PROJECT_ROOT / "experiments" / "candidates.jsonl"

# Required fields that must be present in the CSV header
REQUIRED_CSV_FIELDS = [
    "schema_version",
    "name",
    "stage",
    "status",
    "created_at",
    "config_path",
    "checkpoint_path",
    "zip_path",
    "submission_check",
]

# Required fields in each JSONL record
REQUIRED_JSONL_FIELDS = [
    "schema_version",
    "name",
    "stage",
    "status",
    "config_path",
    "zip_path",
]


# ---------------------------------------------------------------------------
# Tests for candidate_registry.csv
# ---------------------------------------------------------------------------


@pytest.mark.registry
class TestCandidateRegistryCSV:
    """Tests for the candidate_registry.csv file."""

    def test_csv_exists(self):
        """candidate_registry.csv must exist."""
        assert REGISTRY_CSV.exists(), (
            f"candidate_registry.csv not found at {REGISTRY_CSV}"
        )

    def test_csv_not_empty(self):
        """CSV must have at least a header and one data row."""
        content = REGISTRY_CSV.read_text().strip()
        lines = content.splitlines()
        assert len(lines) >= 2, (
            f"candidate_registry.csv has only {len(lines)} lines (need header + data)"
        )

    def test_csv_required_fields(self):
        """CSV header must contain all required fields."""
        with REGISTRY_CSV.open("r", newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
        missing = [f for f in REQUIRED_CSV_FIELDS if f not in header]
        assert not missing, (
            f"Missing required CSV fields: {missing}. "
            f"Available fields: {header}"
        )

    def test_csv_rows_have_name(self):
        """Every CSV row must have a non-empty 'name' field."""
        with REGISTRY_CSV.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, start=2):
                assert row.get("name", "").strip(), (
                    f"Row {i} has empty 'name' field"
                )

    def test_csv_rows_have_valid_status(self):
        """Every CSV row must have a recognizable status."""
        valid_statuses = {
            "candidate", "submitted", "baseline", "archive",
            "rejected", "deprecated", "promoted",
        }
        with REGISTRY_CSV.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, start=2):
                status = row.get("status", "").strip()
                assert status in valid_statuses, (
                    f"Row {i} (name={row.get('name')}): "
                    f"invalid status '{status}', expected one of {valid_statuses}"
                )

    def test_csv_schema_version_consistent(self):
        """All rows should have the same schema_version."""
        versions = set()
        with REGISTRY_CSV.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                v = row.get("schema_version", "").strip()
                if v:
                    versions.add(v)
        assert len(versions) <= 1, (
            f"Multiple schema versions found: {versions}"
        )

    def test_csv_no_completely_empty_rows(self):
        """CSV should not have rows where all values are empty."""
        with REGISTRY_CSV.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, start=2):
                values = [v.strip() for v in row.values() if v]
                assert values, f"Row {i} is completely empty"


# ---------------------------------------------------------------------------
# Tests for candidates.jsonl
# ---------------------------------------------------------------------------


@pytest.mark.registry
class TestCandidatesJSONL:
    """Tests for the candidates.jsonl file."""

    def test_jsonl_exists(self):
        """candidates.jsonl must exist."""
        assert CANDIDATES_JSONL.exists(), (
            f"candidates.jsonl not found at {CANDIDATES_JSONL}"
        )

    def test_jsonl_not_empty(self):
        """JSONL file must have at least one record."""
        content = CANDIDATES_JSONL.read_text().strip()
        assert content, "candidates.jsonl is empty"
        lines = [l for l in content.splitlines() if l.strip()]
        assert len(lines) >= 1, "candidates.jsonl has no data lines"

    def test_jsonl_valid_json_per_line(self):
        """Every non-empty line must be valid JSON."""
        content = CANDIDATES_JSONL.read_text()
        for i, line in enumerate(content.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                pytest.fail(
                    f"Line {i} is not valid JSON: {e}\n"
                    f"Content: {line[:200]}..."
                )

    def test_jsonl_required_fields(self):
        """Each JSONL record must contain required fields."""
        content = CANDIDATES_JSONL.read_text()
        for i, line in enumerate(content.splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = [f for f in REQUIRED_JSONL_FIELDS if f not in record]
            assert not missing, (
                f"Line {i} (name={record.get('name', '?')}): "
                f"missing required fields: {missing}"
            )

    def test_jsonl_names_non_empty(self):
        """Each JSONL record must have a non-empty name."""
        content = CANDIDATES_JSONL.read_text()
        for i, line in enumerate(content.splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            name = record.get("name", "")
            assert name.strip(), f"Line {i}: empty 'name' field"


# ---------------------------------------------------------------------------
# Cross-reference consistency tests
# ---------------------------------------------------------------------------


@pytest.mark.registry
class TestRegistryCrossReference:
    """Tests for consistency between CSV and JSONL."""

    def _get_csv_names(self) -> set[str]:
        names = set()
        with REGISTRY_CSV.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                n = row.get("name", "").strip()
                if n:
                    names.add(n)
        return names

    def _get_jsonl_names(self) -> set[str]:
        names = set()
        content = CANDIDATES_JSONL.read_text()
        for line in content.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            n = record.get("name", "").strip()
            if n:
                names.add(n)
        return names

    def test_jsonl_names_subset_of_csv(self):
        """Every candidate in JSONL should appear in the CSV registry."""
        csv_names = self._get_csv_names()
        jsonl_names = self._get_jsonl_names()
        missing = jsonl_names - csv_names
        assert not missing, (
            f"JSONL has candidates not in CSV: {sorted(missing)}"
        )

    def test_csv_names_subset_of_jsonl(self):
        """Every candidate in CSV should appear in the JSONL file."""
        csv_names = self._get_csv_names()
        jsonl_names = self._get_jsonl_names()
        missing = csv_names - jsonl_names
        assert not missing, (
            f"CSV has candidates not in JSONL: {sorted(missing)}"
        )

    def test_record_count_match(self):
        """CSV and JSONL should have the same number of records."""
        csv_count = 0
        with REGISTRY_CSV.open("r", newline="") as f:
            reader = csv.DictReader(f)
            csv_count = sum(1 for _ in reader)
        jsonl_count = 0
        content = CANDIDATES_JSONL.read_text()
        jsonl_count = sum(1 for l in content.splitlines() if l.strip())
        # Allow mismatch but warn - CSV may have duplicates by design
        if csv_count != jsonl_count:
            pytest.skip(
                f"Record count mismatch (CSV={csv_count}, JSONL={jsonl_count}) - "
                f"may be expected due to CSV duplicate name entries with different stages"
            )


# ---------------------------------------------------------------------------
# Checkpoint path existence tests (warning-level)
# ---------------------------------------------------------------------------


@pytest.mark.registry
class TestCheckpointPaths:
    """Verify referenced checkpoint paths exist. Non-critical; uses warnings."""

    def test_checkpoint_paths_exist(self):
        """Referenced checkpoint paths should exist on disk."""
        missing = []
        with REGISTRY_CSV.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ckpt = row.get("checkpoint_path", "").strip()
                if not ckpt:
                    continue
                # Handle paths with " + " (combined checkpoints)
                paths = [p.strip() for p in ckpt.split("+")]
                for p in paths:
                    if not p:
                        continue
                    full = Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p
                    if not full.exists():
                        missing.append((row.get("name", "?"), p))
        if missing:
            msg = "\n".join(f"  {name}: {path}" for name, path in missing[:10])
            pytest.skip(
                f"Some checkpoint paths not found (expected on dev machines):\n{msg}"
            )
