# ProFE-Net

ProFE-Net predicts mutation effects on protein-complex binding. It extends
[RDE-PPI](https://github.com/luost26/RDE-PPI)
([paper](https://openreview.net/forum?id=_X9Yl1K2mD), ICLR 2023) and includes an
RDE reference model.

## Weights

Copy the ProFE-Net and RDE files from `LmPro-Nano_Zenodo.zip` into `weights/`.

| File | Model |
|---|---|
| `profenet_v3_backbone.pt` | ProFE-Net v3 pretrained backbone |
| `profenet_v3_table3_fold{0,1,2}.pt` | ProFE-Net v3 fold models |
| `rde_backbone.pt` | RDE pretrained backbone |
| `rde_table3_fold{0,1,2}.pt` | RDE fold models |

## Predict mutation effects

Run from `profenet/` in the
[ProFE-Net environment](../README.md#profe-net-mutation-scoring):

```bash
python examples/predict_ddg.py \
  --pdb /path/to/complex.pdb --ligand-chains H --receptor-chains A \
  --mutations YH100W
```

Use `examples/predict_ddg_rde.py` for RDE. Outputs are per-fold and mean ΔΔG
(kcal/mol); negative values predict stronger binding. See the
[root quick start](../README.md#quick-start-mutation-effect-scoring) for mutation
syntax and [evaluation notes](docs/EVALUATION.md) for metrics and split protocols.

## Data download and construction

Run from `profenet/`. Downloads require `wget`, `curl` and `rsync`.

### 1. SKEMPI

Download the [CSV and PDB archive](https://life.bsc.es/pid/skempi2/database/index):

```bash
(cd data && bash get_skempi_v2.sh)
```

Expected paths: `data/SKEMPI_v2/skempi_v2.csv` and `data/SKEMPI_v2/PDBs/*.pdb`.
The dataset loader creates `entries.pkl` and `structures.pkl` in its configured
cache directory. Use a new cache when changing inputs or filters.

### 2. PDB-REDO

Download final mmCIF coordinates from the
[rsync service](https://pdb-redo.eu/download):

```bash
mkdir -p data/PDB_REDO
rsync -av --prune-empty-dirs \
  --exclude='attic/' \
  --include='*/' --include='*_final.cif' --exclude='*' \
  rsync://rsync.pdb-redo.eu/pdb-redo/ data/PDB_REDO/
```

Keep the layout `data/PDB_REDO/<middle-two>/<pdb>/<pdb>_final.cif`.
For a new build, download the RDE-PPI cluster and split lists:

```bash
curl --fail --location \
  https://raw.githubusercontent.com/luost26/RDE-PPI/main/data/pdbredo_clusters.txt \
  --output data/pdbredo_clusters.txt
curl --fail --location \
  https://raw.githubusercontent.com/luost26/RDE-PPI/main/data/pdbredo_splits.txt \
  --output data/pdbredo_splits.txt
```

With SKEMPI inputs in place, remove overlapping clusters:

```bash
python scripts/build_pdbredo_skempi_disjoint.py \
  --output-dir data/generated/pdbredo_skempi50_disjoint
```

Build the training structure cache:

```bash
python - <<'PY'
from rde.datasets.pdbredo_chain import PDBRedoChainDataset

dataset = PDBRedoChainDataset(
    split="train",
    pdbredo_dir="data/PDB_REDO",
    clusters_path="data/generated/pdbredo_skempi50_disjoint/pdbredo_clusters.txt",
    splits_path="data/generated/pdbredo_skempi50_disjoint/pdbredo_splits.txt",
    processed_dir="data/pdbredo_processed_final",
    num_preprocess_jobs=8,
    cache_namespace="skempi50_disjoint_v1",
)
print("Training chains:", len(dataset))
PY
```

Outputs include `structures.lmdb` and `keys.pkl`. Reproducing an earlier dataset
requires its original structures, cluster lists and splits.

### 3. Optional per-chain features

```bash
python -m pip install scipy==1.10.1 scikit-learn==1.3.2
python preprocess_features_skempi.py \
  configs/preprocess/skempi_chain_features.yml --num_workers 8
```

Edit the configuration paths as needed; output defaults to
`data/skempi_chain_features/`. The standalone [create_lmdb.py](create_lmdb.py)
uses integer keys and cannot replace the PDB-keyed cache above.

## License and citation

[Apache-2.0](LICENSE), inherited from RDE-PPI. Cite the paper linked above and the
[LmPro-Nano manuscript](../README.md#citation).
