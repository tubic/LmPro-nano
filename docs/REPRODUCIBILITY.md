# Training and evaluation

Run commands from the repository root. Installation and data preparation are in
[README.md](../README.md). The model uses the dictionaries in `data/` and the
atom14 residue order in [ab_dictionary.py](../fairseq_models/data/ab_dictionary.py).

## Environment

The released model uses Python 3.10.20, PyTorch 2.6.0 with CUDA 12.4 and the bundled
fairseq 0.10.2. Dependencies are listed in
[requirements-main.txt](../environment/requirements-main.txt); the recorded
package versions are in [main-pip-freeze.txt](../environment/main-pip-freeze.txt).
Use the separate ProFE-Net environment for mutation scoring.

## Data

Pretraining reads paired residue/tag binaries from `abgnn/final_3_data/{seq,tag}/`.
Fine-tuning reads `{train,valid,test}_data.jsonl` under
`abgnn/finetune/exp2-hern/<split>/`. Coordinates must follow sequence order and
have shape `[length, 14, 3]`; CDR tags use `0`, `1`, `2` and `3`.

[make_clustered_split.py](../tools/make_clustered_split.py) groups records sharing
a PDB identifier or CDR-H3 sequence cluster before assigning groups to partitions.
Defaults are 60% CDR-H3 identity, 80% coverage, an 80/10/10 split and seed 42.
`--antigen-disjoint` also groups antigen sequences at 40% identity and 60% coverage.
The script checks overlap across the resulting partitions.

Pretraining overlap is a separate check:
[audit_pretrain_overlap.py](../tools/audit_pretrain_overlap.py) checks exact
sequence/CDR-H3 matches; [audit_pretrain_homology.py](../tools/audit_pretrain_homology.py)
checks sequence similarity. Evaluation subset files contain zero-based JSONL
line indices and depend on the original record order.

## Pretraining

[run_pretrain_corrected.sh](../run_pretrain_corrected.sh) sets these defaults and
calls [pretrain-abbert.sh](../pretrain-abbert.sh):

| Setting | Default |
|---|---|
| Task / criterion | `antibody_masked_lm` / `antibody_masked_lm` |
| Updates / warmup / peak learning rate | 100,000 / 10,000 / 0.0006 |
| Optimizer / weight decay | Adam, betas 0.9/0.98, epsilon 1e-6 / 0.01 |
| Sequence mode / maximum length | `eos` / 256 tokens |
| Micro-batch / target effective batch | 112 per GPU / 4,032 sequences |
| Mask probability / full-CDR3 probability | 0.5 / 0.5 |
| Context mode | `keep_unmasked` |
| Seed / checkpoint interval | 1 / 500 updates |

Masking uses the CDR tags; framework residues remain visible. Selected residues
are replaced by `<mask>`. Validation/test masks are fixed. The launcher adjusts
gradient accumulation to the selected GPU count. Set `CUDA_VISIBLE_DEVICES` to
choose GPUs and override environment variables to change the schedule.

```bash
bash run_pretrain_corrected.sh
```

## Fine-tuning

[run_finetune_corrected_exp2.sh](../run_finetune_corrected_exp2.sh) sets the
following values before calling `run_finetune_exp2.sh`:

| Setting | Default |
|---|---|
| Task / criterion | `antibody_generation_task` / `antibody_generation_loss` |
| Learning rate / scheduler | 0.00003 / fixed |
| Micro-batch / accumulation | 4 / 2 |
| Maximum epochs / validation-loss patience | 120 / 40 |
| Optimizer / weight decay / gradient clipping | Adam / 0.01 / 10.0 |
| Refinement cycles / gradient depth | 5 / all cycles (`BPTT_ITERATIONS=-1`) |
| Encoder adaptation / prefix length | frozen encoder with prefix tuning / 5 |
| Encoder sequence / decoder sequence / coordinate loss weights | 1 / 1 / 0.02 |
| Loss normalization / embedding normalization | `token_mean` / `layernorm` |
| Seed / split | 128 / `nanobody_clustered` |

```bash
LMPRO_ENV_BIN="$PWD/.venv/bin" \
  PRETRAINED_FILE=checkpoints/release/lm_nano_pretrain_u33000.pt \
  bash run_finetune_corrected_exp2.sh
```

Set `LMPRO_ENV_BIN` to the installed environment's `bin` directory. `SEED` and
`SPLIT` select the run seed and data directory. Early stopping uses validation
loss; validation-AAR checkpoint comparisons are handled by
[select_finetune_candidate.py](../tools/select_finetune_candidate.py).

## Inference

[inference.py](../inference.py) processes one complex at a time. `--num_decode`
sets the number of sampled sequences; `--topk` retains those with lowest
perplexity. `--report_topks` evaluates several cutoffs on the same samples.
Perplexity is `exp(mean token negative log-likelihood)`, and AAR is the fraction
of recovered residues across retained sequences. Sequence samples share the
structure generated for that complex.

| Output field | Definition |
|---|---|
| `local_cdr3_ca_rmsd` | CDR-H3 C-alpha RMSD after a local Kabsch fit |
| `framework_anchored_cdr3_ca_rmsd` | C-alpha RMSD in the supplied framework frame, without fitting |
| `framework_anchored_cdr3_backbone_rmsd` | N/CA/C RMSD in that frame, without fitting |

`--per_record_out` includes the source JSONL line index. `--restrict_records`
selects an external record list; `--seed` fixes sampling.

## Baseline inputs

[export_baseline_splits.py](../tools/export_baseline_splits.py) exports PDB
assignments and RefineGNN/MEAN input formats. MEAN export also needs its summary
and structure directories. For comparisons, use the same records, candidate
counts, selection rule and coordinate inputs across methods.

See [model implementation](MODIFICATIONS_VS_ABGNN.md),
[CDR-H3 resampling](OPTIMIZATION_ALGORITHM.md) and
[mutation-effect evaluation](../profenet/docs/EVALUATION.md) for the other model paths.
