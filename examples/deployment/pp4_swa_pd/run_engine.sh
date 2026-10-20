#!/usr/bin/env bash
# Start one side of the PP4 prefill / DP4 decode deployment.
set -Eeuo pipefail

role=${1:?usage: run_engine.sh p|d}
case "$role" in p|d) ;; *) echo "role must be p or d" >&2; exit 2 ;; esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VLLM_SOURCE_DIR=${VLLM_SOURCE_DIR:-$(realpath "$script_dir/../../..")}
VLLM_INSTALL_DIR=${VLLM_INSTALL_DIR:-/usr/local/lib/python3.12/dist-packages}
IMAGE=${IMAGE:-docker.io/vllm/vllm-openai:v0.31.0}
CONTAINERD_NAMESPACE=${CONTAINERD_NAMESPACE:-default}
CONTAINER_PREFIX=${CONTAINER_PREFIX:-pp4-swa-pd}
P_PORT=${P_PORT:-8111}
D_PORT=${D_PORT:-8112}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8999}
P_GPU_IDS=${P_GPU_IDS:-0,1,2,3}
D_GPU_IDS=${D_GPU_IDS:-4,5,6,7}
PP_LAYER_PARTITION=${PP_LAYER_PARTITION:-22,20,20,16}
PREFIX_RETENTION_INTERVAL=${PREFIX_RETENTION_INTERVAL:-4096}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1048576}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the absolute target weight directory}
DRAFT_DIR=${DRAFT_DIR:?set DRAFT_DIR to the absolute draft weight directory}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:?set SERVED_MODEL_NAME}
RDMA_DEVICES=${RDMA_DEVICES:?set RDMA_DEVICES to comma-separated RoCE devices}

for dir in "$VLLM_SOURCE_DIR" "$MODEL_DIR" "$DRAFT_DIR"; do
  [[ "$dir" = /* && -d "$dir" ]] || {
    echo "Expected an existing absolute directory: $dir" >&2
    exit 1
  }
done

mounts=()
add_source_mount() {
  local relative=$1
  local source="$VLLM_SOURCE_DIR/$relative"
  [[ -r "$source" ]] || { echo "Missing source: $source" >&2; exit 1; }
  mounts+=(--mount "type=bind,src=$source,dst=$VLLM_INSTALL_DIR/$relative,options=rbind:ro")
}
add_source_mount vllm/config/vllm.py
add_source_mount vllm/v1/core/kv_cache_utils.py
add_source_mount vllm/v1/core/sched/scheduler.py
add_source_mount vllm/models/deepseek_v32/nvidia/model.py
add_source_mount vllm/model_executor/models/deepseek_v2.py
add_source_mount vllm/v1/worker/gpu/pp_utils.py

spec=$(DRAFT_DIR="$DRAFT_DIR" python3 -c 'import json,os; print(json.dumps({"method":"dspark","model":os.environ["DRAFT_DIR"],"num_speculative_tokens":8,"draft_tensor_parallel_size":1,"draft_sample_method":"greedy","rejection_sample_method":"block","enable_adaptive_verification":False,"attention_backend":"FLASH_ATTN","kv_cache_dtype":"fp8"}))')

if [[ "$role" == p ]]; then
  port=$P_PORT
  gpu_ids=$P_GPU_IDS
  kv_role=kv_producer
  parallel=(--tensor-parallel-size 1 --pipeline-parallel-size 4 --distributed-executor-backend mp)
  role_args=(--gpu-memory-utilization 0.90 --max-num-batched-tokens 32768 --max-num-seqs 8
    --safetensors-load-strategy lazy --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}')
  envs=(--env "VLLM_PP_LAYER_PARTITION=$PP_LAYER_PARTITION"
    --env "VLLM_MOONCAKE_BOOTSTRAP_PORT=$BOOTSTRAP_PORT"
    --env VLLM_KV_CACHE_LAYOUT=BLHNC)
else
  port=$D_PORT
  gpu_ids=$D_GPU_IDS
  kv_role=kv_consumer
  parallel=(--tensor-parallel-size 1 --data-parallel-size 4
    --data-parallel-backend mp --data-parallel-size-local 4
    --enable-expert-parallel --distributed-executor-backend mp)
  role_args=(--max-num-seqs 32 --max-num-batched-tokens 128
    --gpu-memory-utilization 0.94
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}')
  envs=(--env VLLM_KV_CACHE_LAYOUT=BLHNC --env VLLM_SWA_PD_GROUP_SPLIT=2)
  add_source_mount vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py
  add_source_mount vllm/v1/worker/utils.py
fi

moon=$(RDMA_DEVICES="$RDMA_DEVICES" KV_ROLE="$kv_role" python3 -c 'import json,os; print(json.dumps({"kv_connector":"MooncakeConnector","kv_role":os.environ["KV_ROLE"],"kv_connector_extra_config":{"mooncake_protocol":"rdma","device_name":os.environ["RDMA_DEVICES"]}}))')

IFS=, read -r -a gpu_list <<< "$gpu_ids"
gpu_flags=()
for gpu in "${gpu_list[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU ID: $gpu" >&2; exit 1; }
  gpu_flags+=(--gpus "$gpu")
done

exec ctr -n "$CONTAINERD_NAMESPACE" run --rm --net-host "${gpu_flags[@]}" \
  --device /dev/infiniband --cap-add CAP_SYS_RESOURCE --cap-add CAP_IPC_LOCK \
  "${mounts[@]}" \
  --mount "type=bind,src=/dev/shm,dst=/dev/shm,options=rbind:rw" \
  --mount "type=bind,src=$MODEL_DIR,dst=$MODEL_DIR,options=rbind:ro" \
  --mount "type=bind,src=$DRAFT_DIR,dst=$DRAFT_DIR,options=rbind:ro" \
  --env VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  --env VLLM_HTTP_TIMEOUT_KEEP_ALIVE=75 \
  --env VLLM_SERVER_DEV_MODE=1 \
  --env VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=60 \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env PYTHONHASHSEED=0 \
  "${envs[@]}" "$IMAGE" "$CONTAINER_PREFIX-$role" \
  sh -c 'ulimit -l unlimited; ulimit -n 1048576; exec "$@"' sh \
  vllm serve "$MODEL_DIR" --host 0.0.0.0 --port "$port" \
  --served-model-name "$SERVED_MODEL_NAME" \
  "${parallel[@]}" --block-size 64 --max-model-len "$MAX_MODEL_LEN" \
  "${role_args[@]}" --enable-prefix-caching \
  --prefix-cache-retention-interval "$PREFIX_RETENTION_INTERVAL" \
  --kv-cache-dtype fp8 --speculative-config "$spec" \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  --kv-transfer-config "$moon" \
  --tool-call-parser glm47 --reasoning-parser glm47 --enable-auto-tool-choice \
  --trust-remote-code
