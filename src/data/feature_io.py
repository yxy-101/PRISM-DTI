"""Helpers for loading external protein and drug feature tensors.

The raw EviDTI-style backbone consumes precomputed features instead of running
large ProtTrans, MG-BERT, or GeoGNN encoders inside this project.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

try:
    import numpy as np
except ImportError:  # pragma: no cover - NumPy may be absent in tiny envs.
    np = None

DUMMY_FEATURE_WARNING = (
    "dummy features are only for pipeline debugging, not paper experiments"
)
_DUMMY_WARNING_PRINTED = False
SMILES_ALPHABET = tuple(
    "#%()+-./0123456789=@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz"
)


def resolve_feature_path(path: Union[str, Path]) -> Path:
    """Return a normalized absolute feature path."""
    return Path(path).expanduser().resolve()


def load_feature_tensor(source: Any, feature_name: str) -> torch.Tensor:
    """Load features from a tensor, one path, or a list of paths.

    TODO: Add metadata-aware caching once dataset integration lands.
    """
    if torch.is_tensor(source):
        return source.float()

    if isinstance(source, (str, Path)):
        return _load_single_feature(resolve_feature_path(source), feature_name)

    if isinstance(source, (list, tuple)):
        tensors = [load_feature_tensor(item, feature_name) for item in source]
        return torch.stack(tensors, dim=0)

    raise TypeError("Unsupported %s source type: %r" % (feature_name, type(source)))


def load_feature_bank(
    path: Optional[Union[str, Path]],
    feature_name: str,
) -> Optional[Dict[str, torch.Tensor]]:
    """Load an ID-to-tensor feature dictionary from a `.pt`/`.pth` file."""
    if path in (None, ""):
        return None
    feature_path = Path(path).expanduser()
    if not feature_path.exists():
        raise FileNotFoundError("Missing %s file: %s" % (feature_name, feature_path))
    loaded = torch.load(str(feature_path), map_location="cpu")
    if isinstance(loaded, dict) and isinstance(loaded.get("features"), dict):
        loaded = loaded["features"]
    if not isinstance(loaded, dict):
        raise TypeError("%s must be a dictionary keyed by ID." % feature_name)

    bank = {}
    for feature_id, feature_value in loaded.items():
        normalized_id = normalize_feature_id(feature_id)
        if normalized_id in {"metadata", "meta", "config"}:
            continue
        tensor = _unwrap_feature_object(
            feature_value,
            feature_name="%s[%s]" % (feature_name, feature_id),
        )
        bank[normalized_id] = (
            tensor if tensor.is_floating_point() else tensor.float()
        ).cpu()
    return bank


def get_feature_from_bank(
    feature_bank: Optional[Dict[str, torch.Tensor]],
    feature_id: str,
) -> Optional[torch.Tensor]:
    """Return a feature tensor from an optional ID feature bank."""
    if feature_bank is None:
        return None
    return feature_bank.get(normalize_feature_id(feature_id))


def normalize_feature_id(feature_id: Any) -> str:
    """Normalize CSV and feature-bank IDs to a shared string key."""
    return str(feature_id).strip()


class IDFeatureStore:
    """Load drug/target features by ID, with deterministic dummy fallbacks."""

    def __init__(
        self,
        protein_feature_path: Optional[Union[str, Path]] = None,
        drug2d_feature_path: Optional[Union[str, Path]] = None,
        drug3d_feature_path: Optional[Union[str, Path]] = None,
        protein_dir: Optional[Union[str, Path]] = None,
        drug2d_dir: Optional[Union[str, Path]] = None,
        drug3d_dir: Optional[Union[str, Path]] = None,
        allow_dummy_features: bool = False,
        allow_simple_drug_features: bool = False,
        allow_simple_drug3d_features: Optional[bool] = None,
        strict_feature_coverage: bool = False,
        use_drug3d: bool = True,
        protein_shape: Sequence[int] = (32, 1024),
        drug2d_shape: Sequence[int] = (24, 768),
        drug3d_shape: Sequence[int] = (16, 300),
    ) -> None:
        """Create an ID-addressable feature loader."""
        self.protein_bank = load_feature_bank(
            protein_feature_path,
            feature_name="protein feature bank",
        )
        self.drug2d_bank = load_feature_bank(
            drug2d_feature_path,
            feature_name="2D drug feature bank",
        )
        self.use_drug3d = bool(use_drug3d)
        self.drug3d_bank = None
        if self.use_drug3d:
            self.drug3d_bank = load_feature_bank(
                drug3d_feature_path,
                feature_name="3D drug feature bank",
            )
        self.protein_dir = _optional_path(protein_dir)
        self.drug2d_dir = _optional_path(drug2d_dir)
        self.drug3d_dir = (
            _optional_path(drug3d_dir) if self.use_drug3d else None
        )
        self.has_drug2d_source = (
            self.drug2d_bank is not None or self.drug2d_dir is not None
        )
        self.has_drug3d_source = (
            self.drug3d_bank is not None or self.drug3d_dir is not None
        )
        self.allow_dummy_features = bool(allow_dummy_features)
        self.allow_simple_drug_features = bool(allow_simple_drug_features)
        self.strict_feature_coverage = bool(strict_feature_coverage)
        if allow_simple_drug3d_features is None:
            allow_simple_drug3d_features = allow_simple_drug_features
        self.allow_simple_drug3d_features = bool(allow_simple_drug3d_features)
        self.protein_shape = tuple(int(dim) for dim in protein_shape)
        self.drug2d_shape = tuple(int(dim) for dim in drug2d_shape)
        self.drug3d_shape = tuple(int(dim) for dim in drug3d_shape)
        self._warned_missing_drug3d_ids = set()

    def load_protein_feature(self, target_id: str) -> torch.Tensor:
        """Load one target/protein feature tensor by target ID."""
        bank_feature = get_feature_from_bank(self.protein_bank, target_id)
        if bank_feature is not None:
            return bank_feature
        if self.protein_bank is not None and not self.allow_dummy_features:
            raise FileNotFoundError(
                "Missing protein feature for target_id %s in configured bank."
                % target_id
            )
        return load_feature_by_id(
            feature_id=target_id,
            feature_name="ProtTrans feature",
            feature_dir=self.protein_dir,
            allow_dummy_features=self.allow_dummy_features,
            dummy_shape=self.protein_shape,
        )

    def load_drug2d_feature(
        self,
        drug_id: str,
        smiles: Optional[str] = None,
    ) -> torch.Tensor:
        """Load one 2D drug feature tensor by drug ID."""
        bank_feature = get_feature_from_bank(self.drug2d_bank, drug_id)
        if bank_feature is not None:
            return bank_feature
        if (
            not self.has_drug2d_source
            and self.allow_simple_drug_features
            and smiles
        ):
            return simple_smiles_feature(smiles, shape=self.drug2d_shape)
        return load_feature_by_id(
            feature_id=drug_id,
            feature_name="MG-BERT feature",
            feature_dir=self.drug2d_dir,
            allow_dummy_features=self.allow_dummy_features,
            dummy_shape=self.drug2d_shape,
        )

    def load_drug3d_feature(
        self,
        drug_id: str,
        smiles: Optional[str] = None,
    ) -> torch.Tensor:
        """Load one 3D drug feature tensor by drug ID."""
        bank_feature = get_feature_from_bank(self.drug3d_bank, drug_id)
        if bank_feature is not None:
            return bank_feature
        if self.drug3d_bank is not None:
            if self.strict_feature_coverage:
                raise FileNotFoundError(
                    "Missing 3D drug feature for drug_id %s in configured bank."
                    % drug_id
                )
            return self._missing_drug3d_zero_fallback(drug_id)
        if (
            not self.has_drug3d_source
            and self.allow_simple_drug3d_features
            and smiles
        ):
            return simple_smiles_feature(smiles, shape=self.drug3d_shape)
        return load_feature_by_id(
            feature_id=drug_id,
            feature_name="GeoGNN feature",
            feature_dir=self.drug3d_dir,
            allow_dummy_features=self.allow_dummy_features,
            dummy_shape=self.drug3d_shape,
        )

    def _missing_drug3d_zero_fallback(self, drug_id: str) -> torch.Tensor:
        """Return a zero 3D feature for missing IDs in a configured 3D bank."""
        normalized_id = normalize_feature_id(drug_id)
        if normalized_id not in self._warned_missing_drug3d_ids:
            warnings.warn(
                "Missing 3D drug feature for drug_id %s in configured bank; "
                "using zero fallback with shape %s."
                % (normalized_id, list(self.drug3d_shape)),
                RuntimeWarning,
            )
            self._warned_missing_drug3d_ids.add(normalized_id)
        return torch.zeros(*self.drug3d_shape, dtype=torch.float32)

    def build_model_batch(
        self,
        records: Union[Dict[str, Sequence[Any]], Sequence[Dict[str, Any]]],
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Materialize model feature tensors from a batch of canonical records."""
        normalized = _records_from_collated(records)
        protein_features = [
            self.load_protein_feature(str(record["target_id"])) for record in normalized
        ]
        drug2d_features = [
            self.load_drug2d_feature(
                str(record["drug_id"]),
                smiles=str(record.get("smiles", "")),
            )
            for record in normalized
        ]
        protein_batch, protein_mask = stack_feature_tensors_with_mask(
            protein_features,
            feature_name="ProtTrans feature",
        )
        drug2d_batch, drug2d_mask = stack_feature_tensors_with_mask(
            drug2d_features,
            feature_name="MG-BERT feature",
        )
        batch = {
            "protein_features": protein_batch,
            "protein_mask": protein_mask,
            "drug2d_features": drug2d_batch,
            "drug2d_mask": drug2d_mask,
        }
        if self.use_drug3d:
            drug3d_features = [
                self.load_drug3d_feature(
                    str(record["drug_id"]),
                    smiles=str(record.get("smiles", "")),
                )
                for record in normalized
            ]
            drug3d_batch, drug3d_mask = stack_feature_tensors_with_mask(
                drug3d_features,
                feature_name="3D drug feature",
            )
            batch["drug3d_features"] = drug3d_batch
            batch["drug3d_mask"] = drug3d_mask
        if device is None:
            return batch
        return {key: value.to(device) for key, value in batch.items()}


def load_feature_by_id(
    feature_id: str,
    feature_name: str,
    feature_dir: Optional[Path] = None,
    allow_dummy_features: bool = False,
    dummy_shape: Sequence[int] = (),
) -> torch.Tensor:
    """Load one feature by ID from disk or create a deterministic dummy tensor."""
    feature_id = normalize_feature_id(feature_id)
    if feature_dir is not None:
        path = find_feature_file(feature_dir=feature_dir, feature_id=feature_id)
        if path is not None:
            return _load_single_feature(path, feature_name)

    if allow_dummy_features:
        warn_dummy_features()
        return deterministic_dummy_feature(
            feature_id=feature_id,
            feature_name=feature_name,
            shape=tuple(int(dim) for dim in dummy_shape),
        )

    if feature_dir is None:
        raise FileNotFoundError(
            "No %s directory configured for ID %s." % (feature_name, feature_id)
        )
    raise FileNotFoundError(
        "Missing %s for ID %s under %s." % (feature_name, feature_id, feature_dir)
    )


def find_feature_file(feature_dir: Path, feature_id: str) -> Optional[Path]:
    """Find a `.pt`, `.pth`, or `.npy` feature file for an ID."""
    base_dir = Path(feature_dir).expanduser()
    candidates = []
    for extension in (".pt", ".pth", ".npy"):
        candidates.append(base_dir / ("%s%s" % (feature_id, extension)))
    safe_id = _safe_filename(feature_id)
    if safe_id != feature_id:
        for extension in (".pt", ".pth", ".npy"):
            candidates.append(base_dir / ("%s%s" % (safe_id, extension)))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def deterministic_dummy_feature(
    feature_id: str,
    feature_name: str,
    shape: Sequence[int],
) -> torch.Tensor:
    """Return a deterministic random tensor keyed by feature ID and type."""
    if not shape:
        raise ValueError("Dummy feature shape is required for %s." % feature_name)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_stable_seed("%s:%s" % (feature_name, feature_id)))
    return torch.randn(*shape, generator=generator)


def simple_smiles_feature(smiles: str, shape: Sequence[int]) -> torch.Tensor:
    """Build a deterministic handcrafted SMILES feature tensor for smoke runs."""
    if len(shape) != 2:
        raise ValueError("Simple SMILES feature shape must be [length, dim].")
    seq_len, feature_dim = (int(shape[0]), int(shape[1]))
    if seq_len <= 0 or feature_dim <= 0:
        raise ValueError("Invalid simple SMILES feature shape: %s." % (shape,))

    text = str(smiles or "")
    vector = torch.zeros(feature_dim, dtype=torch.float32)
    length = max(1, len(text))
    vector[0] = min(length, 256) / 256.0
    vector[1] = sum(char.isalpha() for char in text) / length
    vector[2] = sum(char.isdigit() for char in text) / length
    vector[3] = sum(char in "=#:/\\-" for char in text) / length
    vector[4] = sum(char in "()[]" for char in text) / length

    alphabet_width = min(len(SMILES_ALPHABET), max(0, feature_dim - 5))
    alphabet_index = {char: index for index, char in enumerate(SMILES_ALPHABET)}
    for char in text:
        if char in alphabet_index and alphabet_index[char] < alphabet_width:
            vector[5 + alphabet_index[char]] += 1.0 / length

    hash_start = 5 + alphabet_width
    hash_width = feature_dim - hash_start
    if hash_width > 0:
        for ngram_size in (2, 3):
            if len(text) < ngram_size:
                continue
            denom = max(1, len(text) - ngram_size + 1)
            for index in range(denom):
                ngram = text[index : index + ngram_size]
                bucket = _stable_seed("%d:%s" % (ngram_size, ngram)) % hash_width
                vector[hash_start + bucket] += 1.0 / denom

    rows = vector.unsqueeze(0).repeat(seq_len, 1)
    if feature_dim > 0 and seq_len > 1:
        positions = torch.linspace(0.0, 1.0, steps=seq_len)
        rows[:, -1] = positions
    return rows


def warn_dummy_features() -> None:
    """Print the dummy-feature warning once per process."""
    global _DUMMY_WARNING_PRINTED
    if _DUMMY_WARNING_PRINTED:
        return
    warnings.warn(DUMMY_FEATURE_WARNING, RuntimeWarning)
    _DUMMY_WARNING_PRINTED = True


def stack_feature_tensors(
    tensors: Sequence[torch.Tensor],
    feature_name: str,
) -> torch.Tensor:
    """Stack feature tensors, padding variable sequence/node lengths if needed."""
    if not tensors:
        raise ValueError("Cannot stack empty %s tensors." % feature_name)
    normalized = [tensor.float() for tensor in tensors]
    dims = {tensor.dim() for tensor in normalized}
    if len(dims) != 1:
        raise ValueError("%s tensors must have matching ranks." % feature_name)
    rank = normalized[0].dim()
    if rank == 1:
        return torch.stack(normalized, dim=0)
    if rank == 2:
        return _pad_2d_features(normalized, feature_name=feature_name)
    if rank == 3:
        pooled = [
            tensor.reshape(-1, tensor.size(-1))
            for tensor in normalized
        ]
        return _pad_2d_features(pooled, feature_name=feature_name)
    raise ValueError(
        "%s tensors must be 1D, 2D, or 3D, got rank %d." % (feature_name, rank)
    )


def stack_feature_tensors_with_mask(
    tensors: Sequence[torch.Tensor],
    feature_name: str,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Stack feature tensors and return a valid-position mask when applicable."""
    if not tensors:
        raise ValueError("Cannot stack empty %s tensors." % feature_name)
    ranks = {tensor.dim() for tensor in tensors}
    if len(ranks) != 1:
        raise ValueError("%s tensors must have matching ranks." % feature_name)
    rank = tensors[0].dim()
    if rank == 1:
        normalized = [tensor.unsqueeze(0) for tensor in tensors]
        batch = _pad_2d_features(normalized, feature_name=feature_name)
        mask = torch.ones(
            len(normalized),
            1,
            dtype=torch.bool,
        )
        return batch, mask
    normalized = [
        tensor.reshape(-1, tensor.size(-1)) if tensor.dim() == 3 else tensor
        for tensor in tensors
    ]
    batch = _pad_2d_features(normalized, feature_name=feature_name)
    mask = torch.zeros(
        len(normalized),
        batch.size(1),
        dtype=torch.bool,
    )
    for index, tensor in enumerate(normalized):
        mask[index, : tensor.size(0)] = True
    return batch, mask


def _load_single_feature(path: Path, feature_name: str) -> torch.Tensor:
    """Load one `.pt`, `.pth`, or `.npy` feature file."""
    if not path.exists():
        raise FileNotFoundError("Missing %s file: %s" % (feature_name, path))

    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        loaded = torch.load(str(path), map_location="cpu")
    elif suffix == ".npy":
        if np is None:
            raise ImportError("NumPy is required to read .npy feature files.")
        loaded = np.load(str(path))
    else:
        raise ValueError(
            "Unsupported %s format '%s' for file: %s"
            % (feature_name, suffix, path)
        )

    return _unwrap_feature_object(loaded, feature_name).float()


def _optional_path(path: Optional[Union[str, Path]]) -> Optional[Path]:
    """Normalize optional directory paths."""
    if path in (None, ""):
        return None
    return Path(path).expanduser()


def _records_from_collated(
    records: Union[Dict[str, Sequence[Any]], Sequence[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Normalize either collated dict-of-lists or list-of-dicts batches."""
    if isinstance(records, dict):
        length = len(next(iter(records.values()))) if records else 0
        return [
            {key: values[index] for key, values in records.items()}
            for index in range(length)
        ]
    return list(records)


def _pad_2d_features(
    tensors: Sequence[torch.Tensor],
    feature_name: str,
) -> torch.Tensor:
    """Pad `[length, dim]` feature tensors and stack them into a batch."""
    feature_dim = tensors[0].size(-1)
    if any(tensor.size(-1) != feature_dim for tensor in tensors):
        raise ValueError("%s tensors must share the final dimension." % feature_name)
    max_len = max(int(tensor.size(0)) for tensor in tensors)
    batch = tensors[0].new_zeros(len(tensors), max_len, feature_dim)
    for index, tensor in enumerate(tensors):
        batch[index, : tensor.size(0), :] = tensor
    return batch


def _stable_seed(text: str) -> int:
    """Derive a stable CPU generator seed from text."""
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % (2**31)


def _safe_filename(text: str) -> str:
    """Create a conservative filename fallback for arbitrary IDs."""
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in text)


def _unwrap_feature_object(loaded: Any, feature_name: str) -> torch.Tensor:
    """Extract a tensor payload from common feature containers."""
    if torch.is_tensor(loaded):
        return loaded

    if np is not None and isinstance(loaded, np.ndarray):
        return torch.from_numpy(loaded)

    if isinstance(loaded, dict):
        for key in ("features", "feature", "embeddings", "embedding", "x"):
            if key in loaded:
                return _unwrap_feature_object(loaded[key], feature_name)

    raise TypeError(
        "Could not extract tensor payload from %s object: %r"
        % (feature_name, type(loaded))
    )
