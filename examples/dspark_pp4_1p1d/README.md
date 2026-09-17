# dSpark PP4 prefill + DP4 decode example

This example targets the patched vLLM 0.27.1 branch. It uses eight GPUs on
one host: four for a TP1/PP4 prefill engine and four for TP1/DP4 decode. The
prefill pipeline partition `22,20,20,16` assumes a 78-layer target model.
Both engines use Model Runner V2, dSpark with eight proposed tokens, and draft
TP1. The proxy routes each request through prefill and then a decode replica.

The launcher is a template: it contains no checkpoint locations or machine
addresses. Set the values for your deployment before running it. Use a log
directory outside this checkout and keep the log, model, and credential files
out of commits.

```bash
export TARGET_MODEL=/path/to/target-checkpoint
export DRAFT_MODEL=/path/to/dspark-checkpoint
export NIXL_HOST=your-routable-hostname
export API_BIND_HOST=your-routable-hostname
export API_CONNECT_HOST=your-routable-hostname
export LOG_DIR=/path/outside/checkout/logs

bash examples/dspark_pp4_1p1d/launch_pp4_dp4.sh
```

The launcher defaults to P port 8100, D ports 8200–8203, and proxy port 8000;
override `P_PORT`, `D_PORT`, or `PROXY_PORT` if needed. Set
`PROXY_BIND_HOST` and `PROXY_CONNECT_HOST` if the proxy must listen or be
checked through a different hostname. All engines must use the same patched
version: its NIXL metadata is incompatible with unpatched peers. Initial
startup can spend several minutes compiling GPU kernels.

The target model, dSpark checkpoint, GPU architecture, CUDA libraries, and
NIXL installation must be compatible with the host. In particular, verify
that the checkpoint's draft block size is eight and its verifier corresponds
to the selected target model. The launcher's `--block-size 64` controls the
target KV cache and is distinct from the dSpark checkpoint's block size.

For acceptance statistics, scrape all four decode `/metrics` endpoints and
calculate deltas over the test window. Relevant counters are
`spec_decode_num_drafts_total`, `spec_decode_num_draft_tokens_total`,
`spec_decode_num_accepted_tokens_total`, and
`spec_decode_num_accepted_tokens_per_pos_total`. The draft-token acceptance
rate is accepted tokens divided by proposed tokens. vLLM's mean acceptance
length is `1 + accepted tokens / draft rounds`.
