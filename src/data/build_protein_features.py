#!/usr/bin/env python
"""Build target-id keyed protein feature dictionaries from processed CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_GROUPS = (
    "AVLIM",
    "FYW",
    "KRH",
    "DE",
    "STNQ",
    "CGP",
)
FEATURE_DIM = 1024
PROTTRANS_MODEL_NOT_FOUND = (
    "ProtTrans model not found. Please provide --model_path or use --mode simple."
)


def parse_args() -> argparse.Namespace:
    """Parse protein feature build options."""
    parser = argparse.ArgumentParser(description="Build protein feature dictionary")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. bindingdb.")
    parser.add_argument(
        "--mode",
        choices=("simple", "prottrans"),
        default="simple",
        help="Feature mode. simple is a transitional handcrafted feature; prottrans runs ProtT5.",
    )
    parser.add_argument(
        "--model_name",
        default="Rostlab/prot_t5_xl_uniref50",
        help="Model label kept for metadata; --mode prottrans loads --model_path only.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Local ProtTrans/ProtT5 model directory for --mode prottrans.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument(
        "--representation",
        choices=("pooled", "residue"),
        default="pooled",
        help="Save one mean-pooled token or all non-special residue tokens.",
    )
    parser.add_argument(
        "--storage_dtype",
        choices=("float16", "float32"),
        default="float32",
        help="Storage dtype for saved ProtTrans tensors.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save after this many ProtTrans batches for resume safety.",
    )
    parser.add_argument(
        "--no_fp16",
        action="store_true",
        help="Disable automatic fp16 model weights on CUDA.",
    )
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--valid_csv", default=None)
    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    """Build and save target-id keyed protein features."""
    args = parse_args()
    csv_paths = dataset_csv_paths(args)
    target_sequences, conflicts = load_target_sequences(csv_paths)
    if args.mode == "prottrans":
        try:
            result = build_prottrans_features(
                args=args,
                csv_paths=csv_paths,
                target_sequences=target_sequences,
                conflicts=conflicts,
            )
        except FileNotFoundError as exc:
            if str(exc) == PROTTRANS_MODEL_NOT_FOUND:
                print(PROTTRANS_MODEL_NOT_FOUND, file=sys.stderr)
                raise SystemExit(1) from None
            raise
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    features = {
        target_id: simple_protein_feature(sequence)
        for target_id, sequence in sorted(target_sequences.items())
    }
    output_path = resolve_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, output_path)

    example_shape = list(next(iter(features.values())).shape) if features else []
    result = {
        "status": "ok",
        "dataset": args.dataset,
        "mode": args.mode,
        "input_csvs": [relative_path(path) for path in csv_paths],
        "output": relative_path(output_path),
        "num_targets": len(features),
        "feature_shape": example_shape,
        "sequence_conflict_count": conflicts,
        "note": "simple protein features are transitional, not paper features",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def build_prottrans_features(
    args: argparse.Namespace,
    csv_paths: List[Path],
    target_sequences: Dict[str, str],
    conflicts: int,
) -> Dict[str, object]:
    """Extract ProtTrans/ProtT5 sequence embeddings with resume support."""
    model_path = resolve_model_path(args.model_path)
    transformers = load_transformers()
    tokenizer_cls, model_cls = transformers
    device = resolve_device(args.device)
    output_path = resolve_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features = load_existing_feature_dict(output_path)
    existing_count = len(features)
    pending = [
        (target_id, sequence)
        for target_id, sequence in sorted(target_sequences.items())
        if target_id not in features
    ]

    try:
        tokenizer = tokenizer_cls.from_pretrained(
            str(model_path),
            do_lower_case=False,
            local_files_only=True,
        )
        model = model_cls.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
    except OSError as exc:
        raise FileNotFoundError(PROTTRANS_MODEL_NOT_FOUND) from exc
    use_fp16 = device.type == "cuda" and not args.no_fp16
    if use_fp16:
        model = model.half()
    model = model.to(device)
    model.eval()

    processed = 0
    batch_size = max(1, int(args.batch_size))
    save_every = max(1, int(args.save_every))
    with torch.no_grad():
        for batch_index, batch in enumerate(chunks(pending, batch_size), start=1):
            target_ids = [target_id for target_id, _sequence in batch]
            sequences = [
                prepare_prottrans_sequence(sequence)
                for _target_id, sequence in batch
            ]
            encoded = tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(args.max_length),
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state.detach().float().cpu()
            attention_mask = encoded["attention_mask"].detach().cpu()
            for row_index, target_id in enumerate(target_ids):
                token_count = int(attention_mask[row_index].sum().item())
                sequence_token_count = max(1, token_count - 1)
                residue_tokens = hidden[
                    row_index,
                    :sequence_token_count,
                    :,
                ]
                if args.representation == "pooled":
                    feature = residue_tokens.mean(dim=0, keepdim=True)
                else:
                    feature = residue_tokens
                storage_dtype = (
                    torch.float16
                    if args.storage_dtype == "float16"
                    else torch.float32
                )
                features[target_id] = feature.to(storage_dtype).clone()
            processed += len(batch)
            if batch_index % save_every == 0:
                torch.save(features, output_path)
                print(
                    json.dumps(
                        {
                            "status": "progress",
                            "processed_new_targets": processed,
                            "remaining_targets": len(pending) - processed,
                            "output": relative_path(output_path),
                        },
                        sort_keys=True,
                    )
                )

    torch.save(features, output_path)
    example_shape = list(next(iter(features.values())).shape) if features else []
    return {
        "status": "ok",
        "dataset": args.dataset,
        "mode": args.mode,
        "model_name": args.model_name,
        "model_path": str(model_path),
        "device": str(device),
        "fp16": bool(use_fp16),
        "batch_size": batch_size,
        "max_length": int(args.max_length),
        "representation": args.representation,
        "storage_dtype": args.storage_dtype,
        "input_csvs": [relative_path(path) for path in csv_paths],
        "output": relative_path(output_path),
        "num_targets": len(features),
        "existing_targets_skipped": existing_count,
        "new_targets_processed": processed,
        "feature_shape_example": example_shape,
        "feature_shape_contract": (
            "[1, hidden_dim] for pooled or [residue_count, hidden_dim] "
            "for residue representation"
        ),
        "sequence_conflict_count": conflicts,
    }


def dataset_csv_paths(args: argparse.Namespace) -> List[Path]:
    """Return train/valid/test processed CSV paths."""
    defaults = {
        "train": "data/processed/%s_train.csv" % args.dataset,
        "valid": "data/processed/%s_valid.csv" % args.dataset,
        "test": "data/processed/%s_test.csv" % args.dataset,
    }
    paths = [
        resolve_path(args.train_csv or defaults["train"]),
        resolve_path(args.valid_csv or defaults["valid"]),
        resolve_path(args.test_csv or defaults["test"]),
    ]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError("Processed CSV not found: %s" % path)
    return paths


def load_target_sequences(csv_paths: Iterable[Path]) -> Tuple[Dict[str, str], int]:
    """Collect one protein sequence per target ID across processed splits."""
    target_sequences: Dict[str, str] = {}
    conflicts = 0
    required = {"target_id", "protein_sequence"}
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = [
                column
                for column in required
                if column not in set(reader.fieldnames or [])
            ]
            if missing:
                raise ValueError(
                    "Missing required columns in %s: %s"
                    % (csv_path, ", ".join(missing))
                )
            for row in reader:
                target_id = str(row["target_id"]).strip()
                sequence = str(row["protein_sequence"]).strip()
                if not target_id:
                    continue
                if target_id in target_sequences:
                    if target_sequences[target_id] != sequence:
                        conflicts += 1
                    continue
                target_sequences[target_id] = sequence
    if not target_sequences:
        raise ValueError("No target sequences found in processed CSV files.")
    return target_sequences, conflicts


def simple_protein_feature(sequence: str) -> torch.Tensor:
    """Build a normalized handcrafted protein feature shaped `[1, 1024]`."""
    text = str(sequence or "").upper()
    length = max(1, len(text))
    vector = torch.zeros(FEATURE_DIM, dtype=torch.float32)
    vector[0] = min(length, 4096) / 4096.0
    vector[1] = (length ** 0.5) / (4096 ** 0.5)
    vector[2] = sum(residue not in AMINO_ACIDS for residue in text) / length

    aa_to_index = {residue: index for index, residue in enumerate(AMINO_ACIDS)}
    for residue in text:
        if residue in aa_to_index:
            vector[3 + aa_to_index[residue]] += 1.0 / length

    group_start = 3 + len(AMINO_ACIDS)
    for group_index, group in enumerate(AA_GROUPS):
        vector[group_start + group_index] = (
            sum(residue in group for residue in text) / length
        )

    hash_start = group_start + len(AA_GROUPS)
    hash_width = FEATURE_DIM - hash_start
    for kmer_size in (2, 3):
        if len(text) < kmer_size:
            continue
        denom = len(text) - kmer_size + 1
        for index in range(denom):
            kmer = text[index : index + kmer_size]
            bucket = stable_hash("%d:%s" % (kmer_size, kmer)) % hash_width
            vector[hash_start + bucket] += 1.0 / denom
    return vector.unsqueeze(0)


def load_transformers():
    """Import ProtT5 dependencies lazily so simple mode stays lightweight."""
    try:
        from transformers import T5EncoderModel, T5Tokenizer
    except ImportError as exc:
        raise ImportError(
            "--mode prottrans requires transformers and sentencepiece. "
            "Install them in the active environment before extracting ProtT5 features."
        ) from exc
    return T5Tokenizer, T5EncoderModel


def load_existing_feature_dict(output_path: Path) -> Dict[str, torch.Tensor]:
    """Load an existing output dict for ProtTrans resume support."""
    if not output_path.exists():
        return {}
    loaded = torch.load(str(output_path), map_location="cpu")
    if isinstance(loaded, dict) and isinstance(loaded.get("features"), dict):
        loaded = loaded["features"]
    if not isinstance(loaded, dict):
        raise TypeError(
            "Existing feature file is not an ID-to-tensor dict: %s" % output_path
        )
    features = {}
    for target_id, tensor in loaded.items():
        if target_id in {"metadata", "meta", "config"}:
            continue
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)
        features[str(target_id)] = tensor.cpu()
    return features


def prepare_prottrans_sequence(sequence: str) -> str:
    """Format a protein sequence for ProtT5 tokenization."""
    text = str(sequence or "").upper()
    cleaned = [
        residue if residue in AMINO_ACIDS else "X"
        for residue in text
    ]
    return " ".join(cleaned or ["X"])


def chunks(
    items: Sequence[Tuple[str, str]],
    batch_size: int,
) -> Iterable[List[Tuple[str, str]]]:
    """Yield fixed-size batches from a sequence."""
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def resolve_device(device_name: str) -> torch.device:
    """Resolve a requested extraction device with a clear CUDA error."""
    device = torch.device(str(device_name))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for ProtTrans extraction, but it is unavailable."
        )
    return device


def resolve_model_path(model_path: str) -> Path:
    """Resolve and validate a local ProtTrans model path."""
    if not model_path:
        raise FileNotFoundError(PROTTRANS_MODEL_NOT_FOUND)
    path = resolve_path(model_path)
    if not path.exists():
        raise FileNotFoundError(PROTTRANS_MODEL_NOT_FOUND)
    return path


def stable_hash(text: str) -> int:
    """Return a stable integer hash for feature bucketing."""
    import hashlib

    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def resolve_output_path(args: argparse.Namespace) -> Path:
    """Resolve the output feature dictionary path."""
    if args.output:
        return resolve_path(args.output)
    return ROOT / "data" / "features" / "protein" / (
        "%s_protein_%s.pt" % (args.dataset, args.mode)
    )


def resolve_path(path: str) -> Path:
    """Resolve a path relative to the project root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def relative_path(path: Path) -> str:
    """Return a compact repo-relative path string when possible."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
