# Model implementation

Lm-Nano uses the ABGNN structure-generation framework with a nanobody language
encoder. The MIT notice for the ABGNN code is in [LICENSE](LICENSE).

| Component | Implementation | Source |
|---|---|---|
| Language encoder | Six transformer layers, hidden size 768, FFN size 3072 and 12 attention heads | [ab_roberta_base.py](../fairseq_models/models/ab_roberta_base.py) |
| Position encoding | Rotary embeddings on attention queries and keys; prefix positions use the same coordinate convention | [ab_multihead_attention.py](../fairseq_models/modules/ab_multihead_attention.py) |
| Residue mapping | Encoder/decoder dictionaries and atom14 tables use one residue order | [ab_dictionary.py](../fairseq_models/data/ab_dictionary.py) |
| Sequence head | Residual MLP between sequence message passing and output logits | [ab_decoder.py](../fairseq_models/models/ab_decoder.py) |
| Adaptation | Frozen encoder with a trainable five-token prefix and structure/design modules | [fine-tuning launcher](../run_finetune_corrected_exp2.sh) |
| Refinement | Five iterations with gradients through all iterations in the fine-tuning configuration | [ab_decoder.py](../fairseq_models/models/ab_decoder.py) |
| Losses | Token-mean sequence and coordinate losses; padding excluded from geometry calculations | [loss_normalization.py](../fairseq_models/modules/loss_normalization.py) |

Training settings and output metrics are listed in
[Training and evaluation](REPRODUCIBILITY.md). Training and metric checks are in
[tests/test_training_semantics.py](../tests/test_training_semantics.py).
