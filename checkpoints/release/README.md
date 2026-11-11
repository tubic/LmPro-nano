# Lm-Nano checkpoints

Copy these files from `LmPro-Nano_Zenodo.zip` into `checkpoints/release/`.
Weights are distributed separately from the GitHub source.

| File | Size | Model |
|---|---|---|
| `lm_nano_pretrain_u33000.pt` | 494 MB | Pretrained encoder |
| `lmpro_nano_clustered_seed128_epoch51.pt` | 347 MB | Clustered-split design model, seed 128 |
| `lmpro_nano_antigen_disjoint_v2_seed128_epoch10.pt` | 347 MB | Antigen-disjoint design model, seed 128 |

The 3-2A2-4 generator (351 MB) is at
`checkpoints/exp3/RoPe_RoPeflen5_iter5_loss1_1_2_lr0.0001_bsz2_seed128/checkpoint_best.pt`.
See [ProFE-Net weights](../../profenet/README.md#weights) for mutation-scoring models.
