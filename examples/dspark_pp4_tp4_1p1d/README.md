# dSpark 1P1D: PP4 prefill + TP4 decode (vLLM 0.29.0)

This example runs one prefill engine on GPUs 0–3 with TP1/PP4 and one decode
engine on GPUs 4–7 with TP4/PP1. The engines use NIXL push for KV transfer.
Both use the same target and dSpark checkpoint, eight draft tokens and draft
TP1. The prefill layer split is 22,20,20,16.

Install this branch of vLLM 0.29.0 and its CUDA-compatible NIXL extension in
one virtual environment. The model paths must be available to both engines.
Set a routable address for the P-side NIXL channel; do not use `127.0.0.1` if
P and D are on different hosts. Then run:

```bash
export VLLM_PYTHON=/path/to/venv/bin/python
export MODEL_PATH=/path/to/target-model
export DRAFT_PATH=/path/to/dspark-checkpoint
export NIXL_HOST=your-routable-hostname-or-ip
export LOG_DIR=/path/to/log-directory
bash examples/dspark_pp4_tp4_1p1d/launch.sh all
```

The example serves OpenAI-compatible requests on port 8102 by default. Run
`launch.sh prefill`, `launch.sh decode`, or `launch.sh proxy` to start a single
component. `FRONT_PORT`, `P_PORT`, `D_PORT`, `P_SIDE_PORT`, and `D_SIDE_PORT`
override the defaults. The launcher refuses to start when its own PID file
still points to a live process; inspect the log before restarting.

The script sets max model length 200000, prefix caching on, expert parallel
off, model runner V2, prefill async scheduling off, decode async scheduling on,
prefill max sequences/tokens 8/32768, and decode max sequences/tokens 32/256.
It enables the `glm47` tool parser. It explicitly sets
`enable_adaptive_verification=false`, disabling confidence-based adaptive
verification while retaining ordinary eight-token dSpark speculation. The
checkpoint's confidence-head weights may still be present and loaded.

The included proxy exposes `/health`, `/v1/models`, `/v1/chat/completions`,
`/v1/completions`, and `/metrics`. Wait for its `/health` endpoint to return
HTTP 200, then send a short non-streaming request before load testing. A
healthy endpoint alone does not prove that PP4→TP4 KV transfer works.
