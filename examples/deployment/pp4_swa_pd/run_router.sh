#!/usr/bin/env bash
set -Eeuo pipefail

ROUTER_BIN=${ROUTER_BIN:-vllm-router}
ROUTER_PORT=${ROUTER_PORT:-8110}
P_PORT=${P_PORT:-8111}
D_PORT=${D_PORT:-8112}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8999}

exec "$ROUTER_BIN" --host 0.0.0.0 --port "$ROUTER_PORT" \
  --policy consistent_hash --vllm-pd-disaggregation --kv-connector mooncake \
  --prefill "http://127.0.0.1:$P_PORT" "$BOOTSTRAP_PORT" \
  --decode "http://127.0.0.1:$D_PORT" \
  --intra-node-data-parallel-size 1
