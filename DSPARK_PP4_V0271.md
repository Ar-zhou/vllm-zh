# dSpark PP4 + disaggregated serving on vLLM 0.27.1

This branch is based on official vLLM tag `v0.27.1`. It adapts dSpark
speculative decoding to pipeline-parallel prefill and NIXL-based KV transfer
to data-parallel decode. The official 0.27.1 quantized Markov-head support is
retained.

The changes cover:

- a local dSpark replica on each prefill pipeline rank, with the last rank
  proposing draft tokens and earlier ranks receiving the sampled tokens;
- auxiliary hidden-state transfer between pipeline stages;
- producer/consumer KV-region matching by full layer name, including mixed
  attention layouts and physical cache-block strides;
- scheduler, connector, and worker handling for registration, completion,
  padding, and multi-producer cleanup;
- decode-side MoE profiling workspace sizing; and
- a generic PP4/DP4 launcher and request-routing proxy example.

The patched NIXL metadata protocol must match on the prefill and decode
engines. Do not mix this branch with an unpatched engine or a different patch
series. See `examples/dspark_pp4_1p1d/README.md` for a deployment template.

Only source code and generic examples belong in this branch. Checkpoints,
runtime environments, logs, host addresses, credentials, and benchmark data
must remain outside Git.
