# Mutation-effect evaluation

Predictions use ΔΔG = ΔG(mutant) − ΔG(wild type), in kcal/mol. Negative values
indicate stronger predicted binding. Prediction scripts report three fold values
and their mean.

## Split protocols

[train_table3.py](../train_table3.py) supports `--protocol rde3` and `pdb5`.

| Protocol | Partition |
|---|---|
| `rde3` | Three folds grouped by complex identifier; default seed 2022 |
| `pdb5` | Five folds grouped by PDB identifier and balanced by record count |

Complex grouping can place different complexes from one PDB entry in separate
folds. Keep the protocol and single/multiple-mutation subset fixed when comparing
runs. Split construction is implemented in
[skempi_splits.py](../rde/utils/skempi_splits.py).

## Metrics

[rde/utils/skempi.py](../rde/utils/skempi.py) computes:

| Field | Definition |
|---|---|
| `overall_pearson`, `overall_spearman` | Correlation between predicted and observed ΔΔG |
| `rmse_raw`, `mae_raw` | Errors from the untransformed predictions |
| `rmse`, `mae` | Errors after fitting an affine calibration to the evaluated labels |
| `calibration_slope`, `calibration_intercept` | Fitted calibration parameters |
| `auroc` | AUROC for `ddG > 0`, using predicted ΔΔG as the score |

The calibrated error fields follow the RDE-PPI evaluation convention. They use
labels from the evaluated set; use the raw fields for uncalibrated predictions.

## Uncertainty reports

[summarize_table3_uncertainty.py](../summarize_table3_uncertainty.py) resamples
complexes within each outer fold. Defaults are 2,000 bootstrap replicates, seed
2022 and a 95% interval. A paired comparison requires matching sample IDs.
Prediction CSVs require `sample_id`, `fold`, `complex`, `mutstr`, `num_muts`,
`ddG`, `method` and `ddG_pred`.

```bash
python summarize_table3_uncertainty.py \
  --predictions /path/to/predictions.csv \
  --reference /path/to/reference.csv \
  --mode single --outdir /path/to/report
```

Run from `profenet/` with the statistical dependencies in
[requirements-reviewer.txt](../requirements-reviewer.txt). Prediction files are
external inputs.
