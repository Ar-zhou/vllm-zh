# GLM-5.2 DSpark 1P1D on vLLM 0.30.0

This ports the vLLM 0.29.0 PP4 prefill + TP4 decode example to the
`deploy/glm52-pp8-dspark-v030` source base. One prefill engine runs on GPUs
0–3 with TP1/PP4; one decode engine runs on GPUs 4–7 with TP4/PP1. NIXL push
transfers the KV cache, and the included proxy serves port 8102. Both engines
load the same GLM-5.2 target and DSpark draft checkpoint.

The launcher uses the vLLM 0.30.0 `FLASHMLA_SPARSE` target backend, BF16
target and draft KV caches, eight speculative tokens, and a 200,000-token
maximum length. The prefill partition is `22,20,20,16`. It keeps NIXL's
handshake compatibility check enabled and requires both engines to run this
same source checkout.

Prepare a vLLM 0.30.0 CUDA 13 Python environment with NIXL. Then, from this
checkout, unpack the matching native runtime and launch:

```bash
export GLM52_V030_PYTHON=/path/to/vllm-0.30-venv/bin/python
export GLM52_V030_PIP_BIN=/path/to/vllm-0.30-venv/bin/pip
bash deploy/glm52_pp8_v030/prepare_runtime.sh

export VLLM_PYTHON="$GLM52_V030_PYTHON"
export MODEL_PATH=/path/to/GLM-5.2-target
export DRAFT_PATH=/path/to/DSpark-draft
export NIXL_HOST=your-routable-hostname-or-ip
bash examples/dspark_pp4_tp4_1p1d/launch.sh all
```

`NIXL_HOST` must be reachable by every NIXL worker. Set `FRONT_PORT`,
`P_PORT`, `D_PORT`, `P_SIDE_PORT`, or `D_SIDE_PORT` to override the defaults.
`launch.sh prefill`, `decode`, and `proxy` start individual components.
Logs and PID files default to `.runtime/logs/pd-pp4-tp4/`; inspect these before
restarting. The launcher refuses to start a component whose PID file still
points to a live process.

Validate the proxy's `/health` endpoint, then send a short non-streaming
request through `/v1/chat/completions` and inspect both engine logs for a
successful NIXL transfer. A healthy endpoint alone does not establish that
PP4→TP4 KV transfer is correct. Long-context output, acceptance rate, and
throughput require separate tests.

Decode-side pipeline parallelism remains unsupported by `NixlPushConnector`.
PP with hybrid KV cache layouts (HMA) is also unsupported. This example uses
PP only on prefill and TP only on decode.
