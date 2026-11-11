# CDR-H3 resampling

[scripts/covid-optimize.sh](../scripts/covid-optimize.sh) calls
[covid-optimize_Nb_test.py](../covid-optimize_Nb_test.py). It uses a design model,
a frozen neutralization predictor and three sequence-prior models. The input
sequences, split labels and checkpoints must be supplied at the paths in the launcher.

## Iteration

1. Select a parent antibody from the training set.
2. Generate a batch of CDR-H3 sequences with the parent's framework fixed.
3. Apply the sequence filters and the maximum perplexity threshold across the
   three prior models.
4. Add candidates whose predictor score exceeds the selected parent's stored
   reference score to its candidate pool. Keep the highest-scoring `topk` entries.
5. Select a parent pool and take one supervised gradient step on its entries.

The reference score is initialized from each input parent; it is not updated to
the best candidate in the pool. Scores select candidates and train the generator;
coordinate refinement is not part of this loop.

## Parameters

| Argument | Default |
|---|---|
| `--batch_size` | 100 proposals per selected parent |
| `--epochs` | 10,000 iterations |
| `--lr` | 0.0001 |
| `--topk` | 16; the shell launcher sets 64 |
| `--max_prior_ppl` | 10 |
| `--valid_iter` | 1,000 iterations |

The sequence filter rejects `#`, net charge at or outside -3/+3, `N-X-S/T`
patterns and runs of four identical residues. Candidates preserve CDR-H3 length.
Checkpoints are evaluated at `valid_iter`; the model with the highest validation
score is reloaded before the final evaluation and save.
