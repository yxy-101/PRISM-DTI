"""Batch collation skeleton.

TODO: Collate variable-length protein and molecular features into model batches.
"""

from __future__ import annotations

from typing import Dict, List


def collate_dti_batch(records: List[Dict]) -> Dict[str, List]:
    """Collect a list of records into a simple dictionary of lists."""
    # TODO: Add padding, masks, and tensor conversion.
    batch = {}
    for record in records:
        for key, value in record.items():
            batch.setdefault(key, []).append(value)
    return batch
