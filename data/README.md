# Data and external assets

This upload bundles losslessly xz-compressed BioSNAP/BindingDB random-split source CSVs and the final normalized train/valid/test CSVs for all four benchmarks. Davis/KIBA normalized CSVs are derived model inputs, not original upstream affinity matrices. Their original upstream sources are linked below. No trained checkpoints, predictions or result files are bundled. `expected_assets.json` records asset identities from the existing checkpoint inventory; a recorded hash is a reference, not proof that the asset is distributed in this release.

## Source reconstruction

BioSNAP and BindingDB use TAPB random split files at commit `ac846b1463ecf4a031b84bd6caacb57b25d50d4f`: https://github.com/GaomingL1n/TAPB/tree/ac846b1463ecf4a031b84bd6caacb57b25d50d4f/datasets . Place each dataset's `train_with_id.csv`, `val_with_id.csv`, `test_with_id.csv` under `data/raw/<dataset>/random/`.

```bash
python scripts/prepare_dataset.py --dataset bindingdb
python scripts/prepare_dataset.py --dataset biosnap
python scripts/clean_biosnap_pair_leakage.py --apply
```

BioSNAP cleaning removes four train/evaluation overlapping pairs and three duplicate training rows: 19,224 becomes 19,217. Validation/test retain 2,747/5,493 rows. BindingDB train/valid/test sizes are 34,439/4,920/9,840. Running normalization again would overwrite the cleaned BioSNAP training file.

Davis and KIBA use DeepDTA commit `a546a8433a6822e958f36171c4356ad6f414d623`: https://github.com/hkmztrk/DeepDTA/tree/a546a8433a6822e958f36171c4356ad6f414d623/data . Obtain `Y`, `ligands_can.txt`, `proteins.txt` and the setting1 train/test folds for each dataset. Place them under `data/raw/deepdta/davis/` and `data/raw/deepdta/kiba/`.

```bash
python scripts/prepare_stage8b_davis_from_deepdta.py --deepdta-davis-dir data/raw/deepdta/davis
python scripts/prepare_stage8c_davis_binary_labels.py
python scripts/prepare_stage11b_kiba_binary.py --kiba-root data/raw/deepdta/kiba
```

Davis labels use pKd >= 6 with 22,541/2,505/5,010 rows; KIBA uses score >= 12.1 with 78,836/19,709/19,709 rows. Historical stage names are retained in these reconstruction entrypoints to identify the exact transformations. Do not substitute TAPB Davis random splits for the DeepDTA-derived manuscript split.

The CSV contract is `drug_id,target_id,smiles,protein_sequence,label,split` with binary labels. Additional source-index columns may be present. Data source terms apply separately from the software licence; obtain source files from the cited upstream repositories.

Feature tensors are ID-keyed mappings loaded by `src/data/feature_io.py`. The final paths are in each member config. Residue-level ProtT5 features use dimension 1024, Morgan fingerprints use 2048 bits, multiview fingerprints use 5 × 2048, and the conformer representation uses 16 × 300. Context dictionaries have 8 training-target clusters. No validation/test labels are used to build those dictionaries.

Benchmark input CSVs are losslessly stored in `data/benchmark_inputs.tar.xz`; run `python scripts/unpack_data.py` before training. SHA256 checks verify the expanded files.
