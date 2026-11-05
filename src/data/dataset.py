"""Dataset helpers for canonical PRISM-DTI CSV files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union


CANONICAL_COLUMNS = (
    "drug_id",
    "target_id",
    "smiles",
    "protein_sequence",
    "label",
    "split",
)

OPTIONAL_PASSTHROUGH_COLUMNS = (
    "scaffold_id",
    "scaffold_smiles",
    "scaffold_type",
)


class DTIDataset:
    """In-memory DTI records loaded from the canonical processed CSV schema."""

    def __init__(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        records: Optional[Sequence[Dict]] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        """Load canonical rows or store supplied records for smoke tests."""
        if csv_path is not None and records is not None:
            raise ValueError("Provide either csv_path or records, not both.")
        if csv_path is not None:
            self.records = self._load_csv(Path(csv_path), max_samples=max_samples)
        else:
            loaded_records = list(records or [])
            self.records = _limit_records(loaded_records, max_samples=max_samples)

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self.records)

    def __getitem__(self, index: int) -> Dict:
        """Return one canonical DTI record."""
        return self.records[index]

    def _load_csv(self, csv_path: Path, max_samples: Optional[int]) -> List[Dict]:
        """Read canonical processed CSV records from disk."""
        path = csv_path.expanduser()
        if not path.exists():
            raise FileNotFoundError("Canonical CSV not found: %s" % path)

        records = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _validate_columns(path, reader.fieldnames)
            for row in reader:
                record = {
                    "drug_id": row["drug_id"].strip(),
                    "target_id": row["target_id"].strip(),
                    "smiles": row["smiles"].strip(),
                    "protein_sequence": row["protein_sequence"].strip(),
                    "label": _parse_label(row["label"]),
                }
                for column in OPTIONAL_PASSTHROUGH_COLUMNS:
                    if column in row:
                        record[column] = row[column].strip()
                records.append(record)
                if max_samples is not None and len(records) >= int(max_samples):
                    break
        return records


def _validate_columns(path: Path, columns: Optional[Sequence[str]]) -> None:
    """Raise a clear error if a processed CSV is missing canonical columns."""
    column_set = set(columns or [])
    missing = [column for column in CANONICAL_COLUMNS if column not in column_set]
    if missing:
        raise ValueError(
            "Missing canonical columns in %s: %s" % (path, ", ".join(missing))
        )


def _parse_label(value: object) -> int:
    """Parse labels written as 0/1 or 0.0/1.0 into ints."""
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError as exc:
        raise ValueError("Invalid DTI label: %s" % text) from exc
    if numeric not in (0.0, 1.0):
        raise ValueError("Expected binary DTI label 0/1, got %s." % text)
    return int(numeric)


def _limit_records(records: List[Dict], max_samples: Optional[int]) -> List[Dict]:
    """Apply an optional sample cap to in-memory records."""
    if max_samples is None:
        return records
    return records[: int(max_samples)]
