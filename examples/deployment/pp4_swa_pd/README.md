# PP4 prefill + DP4 decode with a sliding-window draft

This example reproduces a single-node 1P1D deployment on vLLM 0.31:
prefill uses TP1/PP4, decode uses TP1/DP4/EP4, and both use DSpark with
eight speculative tokens. KV transfer uses Mooncake over RoCE. GPU KV uses
FP8, a 64-token block, a 1,048,576-token context limit, and GPU prefix
caching with a 4,096-token sliding-window checkpoint interval.

The script overlays the changed source files from this branch onto the
stock vLLM 0.31 image. The decode-side KV group split in this branch is
required so its Mooncake transfer groups match the PP prefill side.
LMCache is not included in this deployment.

Set these variables before starting either engine:

| Variable | Meaning |
| --- | --- |
| MODEL_DIR | Absolute target weight directory, visible on the host |
| DRAFT_DIR | Absolute draft weight directory, visible on the host |
| SERVED_MODEL_NAME | Public API model name |
| RDMA_DEVICES | Comma-separated RoCE device names |
| IMAGE | vLLM 0.31 image; defaults to docker.io/vllm/vllm-openai:v0.31.0 |
| VLLM_SOURCE_DIR | Checkout of this branch; defaults to this script's repository |
| CONTAINERD_NAMESPACE | Containerd namespace; defaults to default |
| ROUTER_BIN | Path to vllm-router; defaults to the executable on PATH |

Run run_engine.sh p on GPUs 0–3 and run_engine.sh d on GPUs 4–7.
After both /health endpoints return 200 on ports 8111 and 8112,
run run_router.sh. The router listens on port 8110. Use
stop_engine.sh p or stop_engine.sh d for shutdown after stopping the router.
P's PP layer partition defaults to 22,20,20,16. Override GPU lists,
ports, layer partition, context length, container prefix, or cache
interval with the corresponding variables in run_engine.sh.

The P batch limit is 32,768 tokens and eight sequences. D uses 128
batched tokens and 32 sequences per replica. The P and D GPU memory
utilization limits are 0.90 and 0.94. These are the validated launch
settings; actual concurrent capacity depends on weights and hardware.
