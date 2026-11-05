#!/usr/bin/env python
"""Build a target-prior confounder dictionary from training embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional fallback.
    np = None

from src.data.dataset import DTIDataset
from src.data.feature_io import IDFeatureStore


def parse_args() -> argparse.Namespace:
    """Parse confounder dictionary build options."""
    parser = argparse.ArgumentParser(description="Build target confounder dictionary")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--processed_csv", default=None)
    parser.add_argument("--embeddings", default="data/target_embeddings.pt")
    parser.add_argument("--protein_feature_path", default=None)
    parser.add_argument("--target_feature_dir", default=None)
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default="data/confounders",
    )
    parser.add_argument(
        "--num-clusters",
        "--n_clusters",
        dest="num_clusters",
        type=int,
        default=32,
    )
    parser.add_argument("--allow_dummy_features", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    """Load training target embeddings, run KMeans, and save tensors."""
    args = parse_args()
    target_ids = None
    if args.processed_csv:
        embeddings, target_ids = load_processed_target_embeddings(
            processed_csv=args.processed_csv,
            protein_feature_path=args.protein_feature_path,
            target_feature_dir=args.target_feature_dir,
            allow_dummy_features=args.allow_dummy_features,
        )
    else:
        embeddings = load_training_target_embeddings(args.embeddings)
    centers, labels = run_kmeans(
        embeddings=embeddings,
        num_clusters=args.num_clusters,
        seed=args.seed,
        max_iter=args.max_iter,
        tol=args.tol,
    )
    p_ci = estimate_cluster_prior(labels, num_clusters=centers.size(0))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    centers_path = output_dir / "cluster_centers.pt"
    p_ci_path = output_dir / "p_ci.pt"
    assignments_path = output_dir / "target_cluster_assignments.csv"
    torch.save(centers.cpu(), centers_path)
    torch.save(p_ci.cpu(), p_ci_path)
    if target_ids is not None:
        write_target_cluster_assignments(
            output_path=assignments_path,
            target_ids=target_ids,
            labels=labels,
        )

    result = {
        "status": "ok",
        "dataset": args.dataset,
        "processed_csv": args.processed_csv,
        "protein_feature_path": args.protein_feature_path,
        "embeddings": None if args.processed_csv else str(args.embeddings),
        "num_targets": int(embeddings.size(0)),
        "embedding_dim": int(embeddings.size(1)),
        "num_clusters": int(centers.size(0)),
        "cluster_centers": str(centers_path),
        "p_ci": str(p_ci_path),
        "p_ci_sum": round(float(p_ci.sum().item()), 8),
    }
    if target_ids is not None:
        result["target_cluster_assignments"] = str(assignments_path)
    print(json.dumps(result, indent=2, sort_keys=True))


def load_processed_target_embeddings(
    processed_csv: str,
    protein_feature_path: Optional[str],
    target_feature_dir: Optional[str],
    allow_dummy_features: bool,
) -> Tuple[torch.Tensor, List[str]]:
    """Load or synthesize one pooled target embedding per unique target ID."""
    dataset = DTIDataset(csv_path=processed_csv)
    feature_store = IDFeatureStore(
        protein_feature_path=protein_feature_path,
        protein_dir=target_feature_dir,
        allow_dummy_features=allow_dummy_features,
        protein_shape=(32, 1024),
    )
    target_ids = []
    seen = set()
    embeddings = []
    for record in dataset:
        target_id = str(record["target_id"])
        if target_id in seen:
            continue
        seen.add(target_id)
        target_ids.append(target_id)
        feature = feature_store.load_protein_feature(target_id)
        embeddings.append(_pool_target_embedding(feature))
    if not embeddings:
        raise ValueError("No target embeddings could be loaded from %s." % processed_csv)
    return torch.stack(embeddings, dim=0).float(), target_ids


def write_target_cluster_assignments(
    output_path: Path,
    target_ids: List[str],
    labels: torch.Tensor,
) -> None:
    """Write target-to-cluster assignments for inspection."""
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target_id", "cluster_id"])
        writer.writeheader()
        for target_id, label in zip(target_ids, labels.cpu().tolist()):
            writer.writerow({"target_id": target_id, "cluster_id": int(label)})


def load_training_target_embeddings(path: str) -> torch.Tensor:
    """Load training target embeddings from `.pt`, `.pth`, or `.npy`.

    TODO: Accept dataset metadata tables once the data pipeline is finalized.
    """
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError("Training target embedding file not found: %s" % source)

    suffix = source.suffix.lower()
    if suffix in {".pt", ".pth"}:
        loaded = torch.load(str(source), map_location="cpu")
    elif suffix == ".npy":
        if np is None:
            raise ImportError("NumPy is required to read .npy embedding files.")
        loaded = np.load(str(source), allow_pickle=True)
    else:
        raise ValueError("Unsupported embedding format: %s" % suffix)

    embeddings = _unwrap_embeddings(loaded).float()
    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)
    if embeddings.dim() > 2:
        embeddings = embeddings.reshape(embeddings.size(0), -1, embeddings.size(-1))
        embeddings = embeddings.mean(dim=1)
    if embeddings.dim() != 2:
        raise ValueError("Target embeddings must resolve to [num_targets, dim].")
    return embeddings


def run_kmeans(
    embeddings: torch.Tensor,
    num_clusters: int,
    seed: int,
    max_iter: int,
    tol: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run KMeans with sklearn when available, otherwise use a torch fallback."""
    num_clusters = max(1, min(int(num_clusters), int(embeddings.size(0))))
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        return _torch_kmeans(embeddings, num_clusters, seed, max_iter, tol)

    estimator = KMeans(
        n_clusters=num_clusters,
        random_state=seed,
        n_init=10,
        max_iter=max_iter,
        tol=tol,
    )
    labels_np = estimator.fit_predict(embeddings.cpu().numpy())
    centers = torch.from_numpy(estimator.cluster_centers_).float()
    labels = torch.as_tensor(labels_np, dtype=torch.long)
    return centers, labels


def estimate_cluster_prior(labels: torch.Tensor, num_clusters: int) -> torch.Tensor:
    """Estimate `p_ci` as the empirical target-prior mass per cluster."""
    counts = torch.bincount(labels.cpu(), minlength=num_clusters).float()
    total = counts.sum().clamp_min(1.0)
    return counts / total


def _torch_kmeans(
    embeddings: torch.Tensor,
    num_clusters: int,
    seed: int,
    max_iter: int,
    tol: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Small deterministic KMeans fallback for CPU smoke runs."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    data = embeddings.float().cpu()
    perm = torch.randperm(data.size(0), generator=generator)
    centers = data[perm[:num_clusters]].clone()
    labels = torch.zeros(data.size(0), dtype=torch.long)

    for _ in range(max_iter):
        distances = torch.cdist(data, centers)
        labels = torch.argmin(distances, dim=1)
        updated = centers.clone()
        for cluster_id in range(num_clusters):
            members = data[labels == cluster_id]
            if members.numel() > 0:
                updated[cluster_id] = members.mean(dim=0)
        shift = torch.norm(updated - centers, dim=1).max()
        centers = updated
        if float(shift) <= tol:
            break

    return centers, labels


def _unwrap_embeddings(loaded: Any) -> torch.Tensor:
    """Extract a target embedding tensor from common containers."""
    if torch.is_tensor(loaded):
        return loaded

    if np is not None and isinstance(loaded, np.ndarray):
        if loaded.dtype == object:
            loaded = loaded.item() if loaded.shape == () else list(loaded)
        else:
            return torch.from_numpy(loaded)

    if isinstance(loaded, dict):
        for key in (
            "target_embeddings",
            "embeddings",
            "features",
            "target_features",
            "x",
        ):
            if key in loaded:
                return _unwrap_embeddings(loaded[key])
        tensor_values = [value for value in loaded.values() if _is_tensor_like(value)]
        if tensor_values:
            pooled = [_pool_target_embedding(_unwrap_embeddings(value)) for value in tensor_values]
            return torch.stack(pooled, dim=0)

    if isinstance(loaded, (list, tuple)):
        tensors = [_pool_target_embedding(_unwrap_embeddings(value)) for value in loaded]
        return torch.stack(tensors, dim=0)

    raise TypeError("Could not extract target embeddings from %r" % type(loaded))


def _pool_target_embedding(tensor: torch.Tensor) -> torch.Tensor:
    """Pool one target embedding to a single vector."""
    tensor = tensor.float()
    if tensor.dim() == 1:
        return tensor
    return tensor.reshape(-1, tensor.size(-1)).mean(dim=0)


def _is_tensor_like(value: Any) -> bool:
    """Return whether a value can plausibly become an embedding tensor."""
    if torch.is_tensor(value):
        return True
    if np is not None and isinstance(value, np.ndarray):
        return True
    return isinstance(value, (list, tuple))


if __name__ == "__main__":
    main()
