# P reduced draft KV / D full draft KV

This launcher separates the two cache semantics:

- P sets `VLLM_DSPARK_MLA_KV_ABLATION=shared_first_4`. It executes all five
  draft layers, keeps the first four cache tensors, and makes the final layer
  read the nearest retained tensor instead of writing a fifth tensor.
- D sets `VLLM_DSPARK_MLA_KV_ABLATION=baseline`. It executes all five layers
  and creates/writes all five draft KV tensors, including the KV for every new
  generated token.

An approximate alternative is now supported by the launcher: both P and D use
`shared_first_4`. Their draft KV group layouts are identical (four groups), so
the existing Mooncake/LMCache metadata path remains usable. D no longer
reconstructs an independent fifth-layer KV; this is an intentional accuracy
tradeoff. The previous P-reduced/D-full arrangement would require an explicit
reconstruction or a different transfer protocol and is not used by this
launcher.
