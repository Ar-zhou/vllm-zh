#!/usr/bin/env bash
set -euo pipefail

# Set these to local paths and a routable address before running this example.
: "${MODEL_PATH:?Set MODEL_PATH to the target model directory}"
: "${DRAFT_PATH:?Set DRAFT_PATH to the dSpark checkpoint directory}"
: "${NIXL_HOST:?Set NIXL_HOST to the P side-channel address}"
: "${VLLM_PYTHON:?Set VLLM_PYTHON to the vLLM 0.29.0 virtualenv Python}"

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-$EXAMPLE_DIR/logs}"
MODEL_NAME="${MODEL_NAME:-glm-5.2-dspark}"
FRONT_PORT="${FRONT_PORT:-8102}"
P_PORT="${P_PORT:-8112}"
D_PORT="${D_PORT:-8120}"
P_SIDE_PORT="${P_SIDE_PORT:-5911}"
D_SIDE_PORT="${D_SIDE_PORT:-5912}"
ACTION="${1:-all}"
mkdir -p "$LOG_DIR"

SPEC_CONFIG="{\"model\":\"$DRAFT_PATH\",\"method\":\"dspark\",\"num_speculative_tokens\":8,\"draft_tensor_parallel_size\":1,\"draft_sample_method\":\"greedy\",\"attention_backend\":\"FLASH_ATTN\",\"enable_adaptive_verification\":false}"
COMMON_ARGS=(
  serve "$MODEL_PATH" --host 0.0.0.0 --served-model-name "$MODEL_NAME"
  --enable-auto-tool-choice --tool-call-parser glm47
  --enable-prefix-caching --no-enable-expert-parallel
  --max-model-len 200000 --kv-cache-dtype bfloat16 --block-size 64
  --attention-backend FLASHINFER_MLA_SPARSE --no-enable-flashinfer-autotune
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"}'
  --speculative-config "$SPEC_CONFIG"
)

alive() {
  [[ -s "$1" ]] && kill -0 "$(<"$1")" 2>/dev/null
}

wait_health() {
  local port="$1" pid_file="$2"
  for _ in $(seq 1 1800); do
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "Port $port healthy"
      return
    fi
    if ! alive "$pid_file"; then
      echo "Service exited; inspect $LOG_DIR" >&2
      return 1
    fi
    sleep 2
  done
  echo "Timed out waiting for port $port" >&2
  return 1
}

start_prefill() {
  local pid_file="$LOG_DIR/prefill.pid"
  if alive "$pid_file"; then echo 'Prefill already running' >&2; return 1; fi
  nohup env VLLM_USE_V2_MODEL_RUNNER=1 \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    VLLM_PP_LAYER_PARTITION=22,20,20,16 \
    VLLM_NIXL_SIDE_CHANNEL_HOST="$NIXL_HOST" \
    VLLM_NIXL_SIDE_CHANNEL_PORT="$P_SIDE_PORT" \
    "$VLLM_PYTHON" -m vllm.entrypoints.cli.main "${COMMON_ARGS[@]}" \
    --port "$P_PORT" --tensor-parallel-size 1 --pipeline-parallel-size 4 \
    --no-async-scheduling --max-num-batched-tokens 32768 --max-num-seqs 8 \
    --kv-transfer-config '{"kv_connector":"NixlPushConnector","kv_role":"kv_producer","engine_id":"P-pp4","kv_connector_extra_config":{"enforce_handshake_compat":false}}' \
    >"$LOG_DIR/prefill.log" 2>&1 </dev/null &
  echo $! >"$pid_file"
  wait_health "$P_PORT" "$pid_file"
}

start_decode() {
  local pid_file="$LOG_DIR/decode.pid"
  if alive "$pid_file"; then echo 'Decode already running' >&2; return 1; fi
  nohup env VLLM_USE_V2_MODEL_RUNNER=1 \
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
    VLLM_NIXL_SIDE_CHANNEL_HOST="$NIXL_HOST" \
    VLLM_NIXL_SIDE_CHANNEL_PORT="$D_SIDE_PORT" \
    "$VLLM_PYTHON" -m vllm.entrypoints.cli.main "${COMMON_ARGS[@]}" \
    --port "$D_PORT" --tensor-parallel-size 4 --pipeline-parallel-size 1 \
    --async-scheduling --max-num-batched-tokens 256 --max-num-seqs 32 \
    --kv-transfer-config '{"kv_connector":"NixlPushConnector","kv_role":"kv_consumer","engine_id":"D-tp4","kv_connector_extra_config":{"push_registration_timeout":300,"enforce_handshake_compat":false}}' \
    >"$LOG_DIR/decode.log" 2>&1 </dev/null &
  echo $! >"$pid_file"
  wait_health "$D_PORT" "$pid_file"
}

start_proxy() {
  local pid_file="$LOG_DIR/proxy.pid"
  if alive "$pid_file"; then echo 'Proxy already running' >&2; return 1; fi
  nohup "$VLLM_PYTHON" "$EXAMPLE_DIR/pd_proxy.py" \
    --host 0.0.0.0 --port "$FRONT_PORT" \
    --prefill "$NIXL_HOST:$P_PORT" --decode "$NIXL_HOST:$D_PORT" \
    --max-decode-inflight 32 >"$LOG_DIR/proxy.log" 2>&1 </dev/null &
  echo $! >"$pid_file"
  wait_health "$FRONT_PORT" "$pid_file"
}

case "$ACTION" in
  prefill) start_prefill ;;
  decode) start_decode ;;
  proxy) start_proxy ;;
  all) start_prefill; start_decode; start_proxy ;;
  *) echo "Usage: $0 {prefill|decode|proxy|all}" >&2; exit 2 ;;
esac
