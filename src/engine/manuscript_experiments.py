"""Fixed manuscript auxiliary protocols; materialization never starts training."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from src.engine.manuscript_recipes import apply_ablation, configure_common

ROOT = Path(__file__).resolve().parents[2]


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def build_jobs(protocol, datasets, output):
    spec = json.loads((ROOT / "configs/manuscript/protocols.json").read_text())
    jobs = []
    if protocol in spec["stored"]:
        for item in spec["stored"][protocol]:
            if item["dataset"] not in datasets:
                continue
            train = json.loads((ROOT / item["train_config"]).read_text())
            test = json.loads((ROOT / item["test_config"]).read_text())
            jobs.append(dict(item, train=train, test=test, config_provenance="recorded resolved source config"))
        return jobs
    for dataset in datasets:
        for source in spec["source_models"][dataset]:
            base = json.loads((ROOT / source).read_text())
            seed = base["seed"]
            conditions = ("full", "no_causal_loss", "no_cam", "no_backdoor", "no_target_randomization") if protocol == "components" else ("rho0_nonorm", "rho0_norm", "rho025_norm")
            for condition in conditions:
                if protocol == "components":
                    train = copy.deepcopy(base) if condition == "full" else apply_ablation(base, condition)
                    tag = f"{dataset}_component_{condition}_seed{seed}"
                    checkpoint = output / "checkpoints" / (tag + ".pt")
                    prediction = output / "predictions" / (tag + "_valid_predictions.csv")
                    train.update(experiment_name=tag, checkpoint_path=str(checkpoint), output_prediction_path=str(prediction))
                else:
                    train = copy.deepcopy(base)
                    tag = f"{dataset}_resnorm_{condition}_seed{seed}"
                    checkpoint = ROOT / "outputs/checkpoints/residual_layernorm_controls" / (tag + ".pt")
                    prediction = ROOT / "outputs/predictions/residual_layernorm_controls" / (tag + "_valid_predictions.csv")
                    args = argparse.Namespace(epochs=10, batch_size=None, learning_rate=1e-4)
                    configure_common(train, args, seed, tag, resolve(base["checkpoint_path"]), checkpoint,
                                     prediction, .25 if condition == "rho025_norm" else 0., condition != "rho0_nonorm")
                test = copy.deepcopy(train)
                test["valid_csv"] = f"data/processed/{dataset}_test.csv"
                test["experiment_name"] = tag + "_test"
                test["output_prediction_path"] = str(prediction).replace("_valid_predictions.csv", "_test_predictions.csv")
                jobs.append(dict(dataset=dataset, seed=seed, condition=condition, train=train, test=test,
                                 source_config=source, recorded_config=False,
                                 config_provenance="reconstruction from retained original recipe; not authenticated historical resolved file"))
    return jobs


def summarize(jobs, output):
    import numpy as np
    import pandas as pd
    from src.utils.metrics import auroc, auprc, brier_score, ece
    rows = []
    for job in jobs:
        for split, config in (("valid", job["train"]), ("test", job["test"])):
            path = resolve(config["output_prediction_path"])
            frame = pd.read_csv(path, dtype={"drug_id": str, "target_id": str})
            if frame.duplicated(["drug_id", "target_id"]).any() or not frame.label.isin([0, 1]).all():
                raise ValueError("Invalid prediction pairs/labels: " + str(path))
            for branch in ("raw", "do"):
                p = frame["p_" + branch].to_numpy(float)
                if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                    raise ValueError("Invalid probabilities")
                y = frame.label.to_numpy()
                rows.append(dict(dataset=job["dataset"], seed=job["seed"], condition=job["condition"], split=split,
                                 branch=branch, n=len(frame), auroc=auroc(y, p), aupr=auprc(y, p),
                                 brier=brier_score(y, p), ece10=ece(y, p, n_bins=10),
                                 hce=int((((p >= .5) != y) & (np.maximum(p, 1-p) >= .8)).sum()),
                                 hcfp=int(((y == 0) & (p >= .8)).sum()),
                                 prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    table = pd.DataFrame(rows)
    table.to_csv(output / "runs.csv", index=False)
    table.groupby(["dataset", "condition", "split", "branch"])[["auroc", "aupr", "brier", "ece10", "hce", "hcfp"]].agg(["count", "mean", "std"]).to_csv(output / "seed_summary.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("components", "residual", "no3d", "cold-start"), required=True)
    parser.add_argument("--mode", choices=("inventory", "materialize", "preflight", "train", "predict", "summarize"), default="inventory")
    parser.add_argument("--dataset", choices=("all", "bindingdb", "biosnap", "davis", "kiba"), default="all")
    parser.add_argument("--seed", type=int, choices=(7,42,2024), default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = (args.output or ROOT / ("outputs/manuscript_" + args.protocol)).resolve()
    datasets = ("bindingdb", "biosnap", "davis", "kiba") if args.dataset == "all" else (args.dataset,)
    jobs = build_jobs(args.protocol, datasets, output)
    if args.seed is not None:
        jobs = [job for job in jobs if job["seed"] == args.seed]
    if not jobs:
        raise ValueError("No recorded/source-derived jobs for this selection.")
    if args.mode == "inventory":
        print(json.dumps([dict(dataset=j["dataset"],seed=j["seed"],condition=j["condition"],provenance=j["config_provenance"]) for j in jobs], indent=2))
        return
    if args.mode == "materialize":
        if output.exists():
            raise FileExistsError("Preserve existing protocol output: " + str(output))
        output.mkdir(parents=True)
        for number, job in enumerate(jobs):
            for split in ("train", "test"):
                (output / f"job{number}_{split}.json").write_text(json.dumps(job[split], indent=2), encoding="utf-8")
        (output / "jobs.json").write_text(json.dumps(jobs, indent=2), encoding="utf-8")
        print("Materialized", len(jobs), "jobs; no training or inference started.")
        return
    saved = json.loads((output / "jobs.json").read_text(encoding="utf-8"))
    if saved != jobs:
        raise ValueError("Source recipes, selection or output paths changed since materialization.")
    if args.mode == "preflight":
        from src.utils.input_validation import validate_real_inputs
        for job in jobs:
            validate_real_inputs(job["train"])
            validate_real_inputs(job["test"], prediction=True)
            initial = job["train"].get("trainer", {}).get("init_checkpoint")
            if initial and not resolve(initial).is_file():
                raise FileNotFoundError("Required initialization checkpoint: " + initial)
        print("Protocol input paths exist; historical execution is not authenticated by preflight.")
    elif args.mode == "summarize":
        summarize(jobs, output)
    else:
        for number, job in enumerate(jobs):
            checkpoint = resolve(job["train"]["checkpoint_path"])
            if args.mode == "train":
                if checkpoint.exists():
                    raise FileExistsError("Refusing to overwrite checkpoint: " + str(checkpoint))
                subprocess.run([sys.executable,"-B",str(ROOT/"scripts/train.py"),"--config",str(output/f"job{number}_train.json")], cwd=ROOT, check=True)
            else:
                for split in ("train", "test"):
                    if resolve(job[split]["output_prediction_path"]).exists():
                        raise FileExistsError("Refusing to overwrite predictions: " + job[split]["output_prediction_path"])
                    subprocess.run([sys.executable,"-B",str(ROOT/"scripts/predict.py"),"--config",str(output/f"job{number}_{split}.json"),"--checkpoint",str(checkpoint)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
