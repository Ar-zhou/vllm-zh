# Pipeline target with tensor-parallel draft

This experimental path supports a target model with pipeline parallelism and a
separate tensor-parallel draft model spanning the same ranks. The supported
layout is target `TP=1`, target `PP=N`, draft `TP=N`, and `DP=DCP=PCP=1`.

The target keeps its normal pipeline communicator. Draft model-parallel code
uses a separate TP communicator and singleton draft PP coordinators. Each rank
loads and allocates its draft shard. The last target PP rank relays hidden
states, auxiliary states, and sampling state to the draft group. Draft tokens
are proposed across the TP ranks and synchronized back into pipeline request
state.

The PP receive path is split for this mode so ranks can receive sampled tokens
and counts, participate in draft TP collectives, and avoid waiting for a draft
proposal that has not yet been computed. Vocabulary-parallel logits are
collected by the existing logits processor in the active draft TP group.

The implementation changes speculative configuration validation, parallel
group lifecycle, model-runner setup and sampling, PP state relay, and draft
embedding/head sharing. It is experimental and currently has no support for
multiple data or context parallel groups. CUDA graph capture is disabled for
the cross-rank draft path until its collective sequence is graph-safe.

## Validation

The implementation has passed syntax checks and a live multi-GPU smoke request.
A full benchmark and a production-shaped throughput test are still required
before using this path for serving.
