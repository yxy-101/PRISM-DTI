"""Run final member configurations in their required training order."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("bindingdb", "biosnap", "davis", "kiba"))
    parser.add_argument("--mode", required=True, choices=("preflight", "train", "predict", "ensemble", "metrics"))
    parser.add_argument("--protocol", choices=("branch-audit", "performance"), default="branch-audit")
    args = parser.parse_args()
    index_path = "configs/performance/index.json" if args.protocol == "performance" else "configs/final/index.json"
    index = json.loads((ROOT / index_path).read_text(encoding="utf-8"))
    group = index["datasets"][args.dataset]
    prediction_prefix = args.dataset + ("_performance" if args.protocol == "performance" else "")

    def run(script, *arguments):
        subprocess.run([sys.executable, "-B", "scripts/" + script + ".py", *arguments], cwd=ROOT, check=True)

    if args.mode == "preflight":
        from src.utils.input_validation import validate_real_inputs
        for path in group["pretrain_configs"] + group["members"]:
            validate_real_inputs(json.loads((ROOT / path).read_text(encoding="utf-8")))
        print("All input paths exist; historical numerical reproduction is a separate check.")
    elif args.mode == "train":
        for path in group["pretrain_configs"] + group["members"]:
            run("train", "--config", path)
    elif args.mode == "predict":
        for path in group["members"]:
            original = json.loads((ROOT / path).read_text(encoding="utf-8"))
            for split in ("valid", "test"):
                config = dict(original)
                config["valid_csv"] = f"data/processed/{args.dataset}_{split}.csv"
                config["output_prediction_path"] = f"outputs/predictions/{config['experiment_name']}_{split}_predictions.csv"
                target = ROOT / "outputs/resolved_configs" / (config["experiment_name"] + "_" + split + ".json")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(config, indent=2), encoding="utf-8")
                run("predict", "--config", str(target), "--checkpoint", config["checkpoint_path"])
    elif args.mode == "ensemble":
        for split in ("valid", "test"):
            arguments = []
            for path, weight in zip(group["members"], group["member_weights"]):
                config = json.loads((ROOT / path).read_text(encoding="utf-8"))
                arguments += ["--pred-file", f"outputs/predictions/{config['experiment_name']}_{split}_predictions.csv"]
                arguments += ["--weight", str(weight)]
            if len(group["member_weights"]) != len(group["members"]):
                raise ValueError("Incomplete frozen member weights.")
            run("build_prediction_ensemble", *arguments, "--output", f"outputs/predictions/{prediction_prefix}_ensemble_{split}.csv")
    else:
        for split in ("valid", "test"):
            run("report_metrics", "--dataset", args.dataset, "--split", split,
                "--pred_file", f"outputs/predictions/{prediction_prefix}_ensemble_{split}.csv", "--ece_bins", "10",
                "--model_name", args.protocol,
                "--train-csv", f"data/processed/{args.dataset}_train.csv")


if __name__ == "__main__":
    main()
