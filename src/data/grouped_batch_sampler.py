"""Group-aware training batches for entity-disjoint DTI experiments."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Iterator, List, Mapping, Sequence

from torch.utils.data import Sampler


class TargetGroupedBatchSampler(Sampler[List[int]]):
    """Sample balanced positive/negative pairs from several targets per batch.

    The sampler uses training records only. Targets are sampled uniformly, then
    each selected target contributes the same number of positive and negative
    interactions. Sampling with replacement is intentional for targets whose
    minority class has fewer records than the requested per-target quota.
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, object]],
        batch_size: int,
        samples_per_target: int = 4,
        seed: int = 7,
    ) -> None:
        self.batch_size = int(batch_size)
        self.samples_per_target = int(samples_per_target)
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size <= 0:
            raise ValueError("target-grouped batch_size must be positive.")
        if self.samples_per_target < 2:
            raise ValueError("samples_per_target must be at least 2.")
        if self.samples_per_target > self.batch_size:
            raise ValueError("samples_per_target cannot exceed batch_size.")

        grouped: Dict[str, Dict[int, List[int]]] = defaultdict(
            lambda: {0: [], 1: []}
        )
        for index, record in enumerate(records):
            target_id = str(record.get("target_id", "")).strip()
            if not target_id:
                raise ValueError("target-grouped batching requires target_id.")
            label = int(record.get("label", -1))
            if label not in (0, 1):
                raise ValueError("target-grouped batching requires binary labels.")
            grouped[target_id][label].append(index)

        self.groups = {
            target_id: label_indices
            for target_id, label_indices in grouped.items()
            if label_indices[0] and label_indices[1]
        }
        if not self.groups:
            raise ValueError(
                "No training target contains both positive and negative samples."
            )
        self.target_ids = sorted(self.groups)
        self.groups_per_batch = max(
            1, self.batch_size // self.samples_per_target
        )
        self.num_batches = int(math.ceil(len(records) / float(self.batch_size)))

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        positive_count = self.samples_per_target // 2
        negative_count = self.samples_per_target - positive_count

        for _ in range(self.num_batches):
            if len(self.target_ids) >= self.groups_per_batch:
                selected_targets = rng.sample(
                    self.target_ids, self.groups_per_batch
                )
            else:
                selected_targets = [
                    rng.choice(self.target_ids)
                    for _ in range(self.groups_per_batch)
                ]

            batch: List[int] = []
            for target_id in selected_targets:
                members = self.groups[target_id]
                batch.extend(
                    rng.choices(members[1], k=positive_count)
                )
                batch.extend(
                    rng.choices(members[0], k=negative_count)
                )

            while len(batch) < self.batch_size:
                target_id = rng.choice(selected_targets)
                label = rng.choice((0, 1))
                batch.append(rng.choice(self.groups[target_id][label]))
            rng.shuffle(batch)
            yield batch[: self.batch_size]

    def summary(self) -> Dict[str, object]:
        """Return checkpoint-safe sampler metadata."""
        return {
            "enabled": True,
            "eligible_target_count": len(self.target_ids),
            "batch_size": self.batch_size,
            "samples_per_target": self.samples_per_target,
            "targets_per_batch": self.groups_per_batch,
            "batches_per_epoch": self.num_batches,
            "seed": self.seed,
        }
