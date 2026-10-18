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
the cross-rank draft path until its collective sequence is graph-safe. Target
decode graph capture can still be enabled with `FULL_DECODE_ONLY`.

## Throughput configuration

The asynchronous PP scheduler spaces consecutive decode steps for the same
request by the pipeline depth. A small `max_num_seqs` value or equally small
client concurrency leaves most pipeline steps with few runnable requests. Size
both to keep multiple microbatches in flight. For a four-stage pipeline, four
concurrent requests are insufficient to evaluate sustained throughput.

`max_num_batched_tokens` must also accommodate the speculative query slots
reserved for every scheduled request. Increase it together with
`max_num_seqs`, then check the actual scheduled-token histogram and KV-cache
capacity. The cross-rank draft requires all pipeline ranks to synchronize on
each proposal, so adding concurrency alone does not remove that cost.

Use `FULL_DECODE_ONLY` for the target when supported by the model and kernels.
The draft remains eager in this path. Benchmark both modes with identical
requests before changing the deployment setting; eager target execution can
substantially understate this layout's throughput.

## Validation

The implementation has passed syntax checks, multi-GPU request tests, and
short concurrency tests with target decode graph capture. Long-context load
and full accuracy regression testing are still required before using this
path for general serving.
