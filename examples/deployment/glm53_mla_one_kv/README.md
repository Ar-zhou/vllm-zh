# GLM DSpark MLA single-cache ablation

This experiment uses the GLM DSpark MLA implementation on vLLM v0.31.0. Set
`VLLM_DSPARK_MLA_KV_ABLATION` before starting the server:

| Mode | Draft KV caches | Draft decoder layers executed |
| --- | ---: | ---: |
| `baseline` | 5 | 5 |
| `shared_first` | 1 | 5 |
| `first_layer` | 1 | 1 |
| `shared_first_N` (`N=2,3,4`) | N | 5 |
| `first_N_layers` (`N=2,3,4`) | N | N |
| `share_swa_N` (`N=1,2,3`) | 4 | 5 |

In `shared_first_N`, later decoder layers keep their own queries, projections,
feed-forward networks, and attention rules, but read layer N's latent KV. They
do not write KV. In `first_N_layers`, the draft executes the first N decoder
layers and the existing final norm and output heads. The `shared_first` and
`first_layer` modes are the N=1 equivalents. The target model, speculative
verification, and target KV are unchanged.

In `share_swa_N`, sliding-window draft layer N reads the latent KV of layer
N-1 and does not write its own cache. Every other layer keeps its own KV,
including the final full-attention layer 4. All five draft decoder layers
still execute. These modes isolate sharing one sliding-window layer without
replacing the final full-attention cache.

For a comparable serving experiment, use the same target and draft checkpoints,
`num_speculative_tokens`, sampling settings, request bodies, concurrency, and
generation limit for every mode. Disable prefix caching to remove cache-warmth
effects. Check the engine's KV-cache log for the intended number of draft
cache layers.

Prometheus counters give the acceptance statistics:

- `vllm:spec_decode_num_accepted_tokens_total /
  vllm:spec_decode_num_draft_tokens_total` is the token acceptance rate.
- `1 + accepted_tokens / num_drafts` is the mean verified output length per
  draft step, including the target's verification token.
- `vllm:spec_decode_num_accepted_tokens_per_pos_total{position="i"} /
  num_drafts` is the probability that draft position `i` is accepted.

Take counter differences before and after each fixed request batch. The
experiments change draft model behavior and need an output-quality check in
addition to acceptance measurements.

The DP4+EP4 comparison on the v0.31.0 image also needs two small serving fixes
included in this branch: the Cutlass MoE permutation scratch buffer must cover
tokens dispatched from all DP ranks, and the KV block zeroing kernel must run
on the cache tensor's CUDA device. The cache-sharing path uses the target
layer's KV specification when a group contains mixed cache specifications.
