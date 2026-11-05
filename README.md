# PRISM-DTI

Source code for drug鈥搕arget interaction prediction and target-prior confidence analysis. This repository contains model/training/prediction code, fixed configurations and benchmark inputs. No trained model weights, prediction exports, experimental metrics, logs or historical result reports are included. Software license selection is deferred by the authors.

## Environment

Recommended: Linux, Python 3.10, PyTorch 2.0.0 and CUDA 11.8. The manuscript reports a 32 GB GPU; actual memory requirements depend on batch size and feature extraction. Other dependencies (NumPy, pandas, SciPy, scikit-learn, RDKit, Transformers, PyYAML and SentencePiece) are pinned in the supplied files. This is a reconstructed installation recipe, not an archived historical environment.

```bash
conda env create -f environment.yaml
conda activate prism-dti
python scripts/unpack_data.py
```

ProtT5 pretrained weights and generated feature banks must be obtained/built separately. No DTI checkpoint is required to train from scratch; prediction requires a checkpoint produced by training.

## Data and features

`data/README.md` describes source repositories, labels and fixed splits. Included data archive contain benchmark pairs/labels/sequences, not model outputs. Do not re-normalize the supplied cleaned BioSNAP split. Original Davis/KIBA affinity sources are linked, with their final normalized inputs included here.

For each dataset, replacing `biosnap` consistently:

```bash
python scripts/build_protein_features.py --dataset biosnap --mode prottrans --model_path /path/to/prot_t5_xl_uniref50 --representation residue --storage_dtype float16 --max_length 1024 --output data/features/protein/biosnap_protein_prottrans_residue_fp16.pt
python scripts/build_drug_morgan_features.py --dataset biosnap
python scripts/build_drug3d_features_from_evidti.py --dataset biosnap
python scripts/build_confounder_dict.py --processed_csv data/processed/biosnap_train.csv --protein_feature_path data/features/protein/biosnap_protein_prottrans_residue_fp16.pt --num-clusters 8 --seed 7 --output-dir data/confounder/biosnap_prottrans_residue
```

BindingDB/KIBA also require five-view fingerprints; the separate performance protocol requires these for BioSNAP/Davis too:

```bash
python scripts/build_drug_multiview_features.py --dataset bindingdb --output data/features/drug/bindingdb_drug_multiview2048.pt --failed-output outputs/preparation/bindingdb_multiview_failed.csv
```

## Train and predict

Run from the repository root. Replace the dataset with bindingdb, biosnap, davis or kiba. Execute the stages in order:

```bash
python scripts/reproduce.py --dataset biosnap --mode preflight
python scripts/reproduce.py --dataset biosnap --mode train
python scripts/reproduce.py --dataset biosnap --mode predict
python scripts/reproduce.py --dataset biosnap --mode ensemble
python scripts/reproduce.py --dataset biosnap --mode metrics
```

Default `--protocol branch-audit` supplies the calibration/risk analyses (6/3/3/6 members). Section 4.2 performance uses `--protocol performance` (six Raw-priority members per dataset). These protocols use different BioSNAP/Davis model members on identical evaluation pairs. Fixed fingerprint-pair weights are applied within seeds, then seeds are averaged; historical weight-selection provenance remains unverified. Do not interchange their outputs. Missing external features cause an explicit error.

`src/` contains reusable implementations; `scripts/` contains main entrypoints; `configs/` retains final and prerequisite settings. Full historical training-to-table reproduction and the reconstructed CUDA environment have not been independently certified.

Benchmark input CSVs are losslessly stored in `data/benchmark_inputs.tar.xz`; run `python scripts/unpack_data.py` before training. SHA256 checks verify the expanded files.

## Repository layout

- src/: model, data loading, losses, training and prediction implementations.
- scripts/: main runnable commands and input unpacking.
- configs/: fixed final and prerequisite configurations.
- data/: benchmark source notes, checksums and compressed benchmark inputs; no predictions or experimental results.
- environment.yaml and requirements.txt: reconstructed runtime dependencies.

Run commands from the repository root after cloning. The benchmark input archive is data storage only, not a trained checkpoint or experimental result.
