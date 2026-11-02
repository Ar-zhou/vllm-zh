# P reduced draft KV / D full draft KV

This launcher separates the two cache semantics:

- P sets `VLLM_DSPARK_MLA_KV_ABLATION=shared_first_4`. It executes all five
  draft layers, keeps the first four cache tensors, and makes the final layer
  read the nearest retained tensor instead of writing a fifth tensor.
- D sets `VLLM_DSPARK_MLA_KV_ABLATION=baseline`. It executes all five layers
  and creates/writes all five draft KV tensors, including the KV for every new
  generated token.

This is not equivalent to simply starting the existing P/D service with two
different environment values. The P and D draft KV group layouts differ (P has
four groups, D has five). Mooncake/LMCache transfer metadata must therefore
either transfer only the target-model KV and let D rebuild the draft prompt KV,
or carry an explicit P-to-D reconstruction payload. Sending P's four-group
draft cache directly into D's five-group cache is invalid.

`run_pd_p_reuse_d_full.sh` sets the two environment values explicitly, but a
deployment must also use a connector path that excludes the reduced P draft
groups from the P/D transfer. The script is a configuration scaffold until
that metadata path is implemented and tested.
