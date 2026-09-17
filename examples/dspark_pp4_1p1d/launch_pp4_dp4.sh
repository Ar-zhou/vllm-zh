#!/usr/bin/env bash
set -euo pipefail

# Single-host example. Supply model paths, network names and the log location.
: "${TARGET_MODEL:?set TARGET_MODEL to the target checkpoint directory}"
: "${DRAFT_MODEL:?set DRAFT_MODEL to the dSpark checkpoint directory}"
: "${NIXL_HOST:?set NIXL_HOST to a host name reachable by both engines}"
: "${API_BIND_HOST:?set API_BIND_HOST for the engine API listeners}"
: "${API_CONNECT_HOST:?set API_CONNECT_HOST for proxy-to-engine requests}"
: "${LOG_DIR:?set LOG_DIR outside the source checkout}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
P_PORT=${P_PORT:-8100}
D_PORT=${D_PORT:-8200}
PROXY_PORT=${PROXY_PORT:-8000}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-glm-dspark}
PROXY_BIND_HOST=${PROXY_BIND_HOST:-localhost}
PROXY_CONNECT_HOST=${PROXY_CONNECT_HOST:-localhost}

mkdir -p "$LOG_DIR"

wait_health() {
  local port=$1
  local host=$2
  for _ in $(seq 1 1800); do
    if curl -fsS "http://${host}:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for port $port" >&2
  return 1
}

draft_config="{\"model\":\"${DRAFT_MODEL}\",\"method\":\"dspark\",\"num_speculative_tokens\":8,\"draft_tensor_parallel_size\":1,\"draft_sample_method\":\"greedy\",\"attention_backend\":\"FLASH_ATTN\"}"

nohup env \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_PP_LAYER_PARTITION=22,20,20,16 \
  VLLM_NIXL_SIDE_CHANNEL_HOST="$NIXL_HOST" \
  VLLM_NIXL_SIDE_CHANNEL_PORT=5811 \
  vllm serve "$TARGET_MODEL" \
  --host "$API_BIND_HOST" --port "$P_PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --enable-auto-tool-choice --tool-call-parser glm47 \
  --tensor-parallel-size 1 --pipeline-parallel-size 4 \
  --no-async-scheduling --enable-prefix-caching --no-enable-expert-parallel \
  --max-model-len 200000 --max-num-batched-tokens 32768 --max-num-seqs 8 \
  --kv-cache-dtype bfloat16 --block-size 64 \
  --attention-backend FLASHINFER_MLA_SPARSE --no-enable-flashinfer-autotune \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config "$draft_config" \
  --kv-transfer-config '{"kv_connector":"NixlPushConnector","kv_role":"kv_producer","engine_id":"P-example","kv_connector_extra_config":{"enforce_handshake_compat":false}}' \
  >"$LOG_DIR/prefill.log" 2>&1 </dev/null &
echo $! >"$LOG_DIR/prefill.pid"
wait_health "$P_PORT" "$API_CONNECT_HOST"

nohup env \
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_NIXL_SIDE_CHANNEL_HOST="$NIXL_HOST" \
  VLLM_NIXL_SIDE_CHANNEL_PORT=5812 \
  vllm serve "$TARGET_MODEL" \
  --host "$API_BIND_HOST" --port "$D_PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --enable-auto-tool-choice --tool-call-parser glm47 \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --data-parallel-size 4 --data-parallel-size-local 4 \
  --data-parallel-multi-port-external-lb \
  --data-parallel-supervisor-port "$((D_PORT + 9))" \
  --async-scheduling --enable-prefix-caching --no-enable-expert-parallel \
  --max-model-len 200000 --max-num-batched-tokens 256 --max-num-seqs 32 \
  --kv-cache-dtype bfloat16 --block-size 64 \
  --attention-backend FLASHINFER_MLA_SPARSE --no-enable-flashinfer-autotune \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config "$draft_config" \
  --kv-transfer-config '{"kv_connector":"NixlPushConnector","kv_role":"kv_consumer","engine_id":"D-example","kv_connector_extra_config":{"push_registration_timeout":300,"enforce_handshake_compat":false}}' \
  >"$LOG_DIR/decode.log" 2>&1 </dev/null &
echo $! >"$LOG_DIR/decode.pid"
for port in "$D_PORT" "$((D_PORT + 1))" "$((D_PORT + 2))" "$((D_PORT + 3))"; do
  wait_health "$port" "$API_CONNECT_HOST"
done

nohup python "$SCRIPT_DIR/pd_proxy.py" \
  --host "$PROXY_BIND_HOST" --port "$PROXY_PORT" \
  --prefill "$API_CONNECT_HOST:$P_PORT" \
  --decode "$API_CONNECT_HOST:$D_PORT" \
           "$API_CONNECT_HOST:$((D_PORT + 1))" \
           "$API_CONNECT_HOST:$((D_PORT + 2))" \
           "$API_CONNECT_HOST:$((D_PORT + 3))" \
  --max-decode-inflight 32 \
  >"$LOG_DIR/proxy.log" 2>&1 </dev/null &
echo $! >"$LOG_DIR/proxy.pid"
wait_health "$PROXY_PORT" "$PROXY_CONNECT_HOST"
echo "PP4 + DP4 ready on port $PROXY_PORT"
