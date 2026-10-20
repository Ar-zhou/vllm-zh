#!/usr/bin/env bash
set -Eeuo pipefail

role=${1:?usage: run.sh p|d|router}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)

: "${MODEL_PATH:?set MODEL_PATH to the target model directory}"
: "${DRAFT_MODEL_PATH:?set DRAFT_MODEL_PATH to the DSpark draft directory}"
: "${SERVED_MODEL_NAME:?set SERVED_MODEL_NAME}"
: "${MOONCAKE_DEVICE_NAME:?set MOONCAKE_DEVICE_NAME to the RDMA device list}"

image=${VLLM_IMAGE:-m.daocloud.io/docker.io/vllm/vllm-openai:v0.31.0}
namespace=${CONTAINERD_NAMESPACE:-default}
prefill_port=${PREFILL_PORT:-8111}
decode_port=${DECODE_PORT:-8112}
router_port=${ROUTER_PORT:-8110}
bootstrap_port=${MOONCAKE_BOOTSTRAP_PORT:-8998}
max_model_len=${MAX_MODEL_LEN:-1048576}
block_size=${BLOCK_SIZE:-64}
draft_tokens=${DSPARK_TOKENS:-8}
prefix_retention=${PREFIX_CACHE_RETENTION_INTERVAL:-0}

[[ -r "$MODEL_PATH/config.json" && -r "$DRAFT_MODEL_PATH/config.json" ]] || {
  echo 'MODEL_PATH and DRAFT_MODEL_PATH must contain config.json' >&2
  exit 1
}

if [[ "$role" == router ]]; then
  exec "${VLLM_ROUTER_BIN:-vllm-router}" \
    --host 0.0.0.0 --port "$router_port" --policy consistent_hash \
    --vllm-pd-disaggregation --kv-connector mooncake \
    --prefill "http://127.0.0.1:$prefill_port" "$bootstrap_port" \
    --decode "http://127.0.0.1:$decode_port" \
    --intra-node-data-parallel-size 1
fi

[[ "$role" == p || "$role" == d ]] || { echo "unknown role: $role" >&2; exit 2; }

spec=$(python3 - "$DRAFT_MODEL_PATH" "$draft_tokens" <<'PY'
import json
import sys
print(json.dumps({
    "method": "dspark",
    "model": sys.argv[1],
    "num_speculative_tokens": int(sys.argv[2]),
    "draft_tensor_parallel_size": 1,
    "draft_sample_method": "greedy",
    "rejection_sample_method": "block",
    "enable_adaptive_verification": False,
    "attention_backend": "FLASH_ATTN",
    "kv_cache_dtype": "fp8",
}))
PY
)

if [[ "$role" == p ]]; then
  port=$prefill_port
  gpu_ids=${P_GPU_IDS:-0,1,2,3}
  kv_role=kv_producer
  parallel=(--tensor-parallel-size 1 --pipeline-parallel-size 4
            --distributed-executor-backend mp)
  role_args=(--gpu-memory-utilization "${P_GPU_MEMORY_UTILIZATION:-0.90}"
             --max-num-batched-tokens "${P_MAX_NUM_BATCHED_TOKENS:-8192}"
             --max-num-seqs "${P_MAX_NUM_SEQS:-8}"
             --safetensors-load-strategy lazy)
  if [[ -n "${P_KV_CACHE_MEMORY_BYTES:-76G}" ]]; then
    role_args+=(--kv-cache-memory-bytes "${P_KV_CACHE_MEMORY_BYTES:-76G}")
  fi
  envs=(--env "VLLM_PP_LAYER_PARTITION=${PP_LAYER_PARTITION:-22,20,20,16}"
        --env "VLLM_MOONCAKE_BOOTSTRAP_PORT=$bootstrap_port")
  gpu_patches=()
else
  port=$decode_port
  gpu_ids=${D_GPU_IDS:-4,5,6,7}
  kv_role=kv_consumer
  parallel=(--tensor-parallel-size 1 --data-parallel-size 4
            --data-parallel-backend mp --data-parallel-size-local 4
            --enable-expert-parallel --distributed-executor-backend mp)
  role_args=(--gpu-memory-utilization "${D_GPU_MEMORY_UTILIZATION:-0.94}"
             --max-num-batched-tokens "${D_MAX_NUM_BATCHED_TOKENS:-128}"
             --max-num-seqs "${D_MAX_NUM_SEQS:-32}"
             --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}')
  envs=()
  gpu_patches=(
    --mount "type=bind,src=$script_dir/patches/cutlass_moe_dpfix.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py,options=rbind:ro"
    --mount "type=bind,src=$script_dir/patches/worker_utils_dpfix.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/utils.py,options=rbind:ro"
  )
fi

moon=$(python3 - "$kv_role" "$MOONCAKE_DEVICE_NAME" <<'PY'
import json
import sys
print(json.dumps({
    "kv_connector": "MooncakeConnector",
    "kv_role": sys.argv[1],
    "kv_connector_extra_config": {
        "mooncake_protocol": "rdma",
        "device_name": sys.argv[2],
    },
}))
PY
)

IFS=, read -r -a gpu_list <<< "$gpu_ids"
gpu_flags=()
for gpu in "${gpu_list[@]}"; do gpu_flags+=(--gpus "$gpu"); done

served_names=("$SERVED_MODEL_NAME")
if [[ -n "${MODEL_ALIAS:-}" ]]; then served_names+=("$MODEL_ALIAS"); fi

# A checkpoint directory may contain links to weight files stored elsewhere.
# Bind their resolved directories at the same absolute paths in the container.
extra_mounts=()
declare -A mounted_extra=()
for model_dir in "$MODEL_PATH" "$DRAFT_MODEL_PATH"; do
  while IFS= read -r -d '' link; do
    target=$(realpath -e -- "$link") || { echo "broken model link: $link" >&2; exit 1; }
    case "$target" in
      "$MODEL_PATH"/*|"$DRAFT_MODEL_PATH"/*) continue ;;
    esac
    if [[ -d "$target" ]]; then mount_dir=$target; else mount_dir=$(dirname -- "$target"); fi
    if [[ -z "${mounted_extra[$mount_dir]+set}" ]]; then
      extra_mounts+=(--mount "type=bind,src=$mount_dir,dst=$mount_dir,options=rbind:ro")
      mounted_extra[$mount_dir]=1
    fi
  done < <(find "$model_dir" -maxdepth 1 -type l -print0)
done

exec ctr -n "$namespace" run --rm --net-host "${gpu_flags[@]}" \
  --device /dev/infiniband --cap-add CAP_SYS_RESOURCE --cap-add CAP_IPC_LOCK \
  --mount 'type=bind,src=/dev/shm,dst=/dev/shm,options=rbind:rw' \
  --mount "type=bind,src=$MODEL_PATH,dst=$MODEL_PATH,options=rbind:ro" \
  --mount "type=bind,src=$DRAFT_MODEL_PATH,dst=$DRAFT_MODEL_PATH,options=rbind:ro" \
  --mount "type=bind,src=$repo_root/vllm/models/deepseek_v32/nvidia/model.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v32/nvidia/model.py,options=rbind:ro" \
  --mount "type=bind,src=$repo_root/vllm/model_executor/models/deepseek_v2.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py,options=rbind:ro" \
  --mount "type=bind,src=$repo_root/vllm/v1/worker/gpu/pp_utils.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/pp_utils.py,options=rbind:ro" \
  "${extra_mounts[@]}" \
  "${gpu_patches[@]}" \
  --env VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  --env VLLM_HTTP_TIMEOUT_KEEP_ALIVE=75 \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env PYTHONHASHSEED=0 "${envs[@]}" \
  "$image" "pp-dspark-${role}" \
  sh -c 'ulimit -l unlimited; ulimit -n 1048576; exec "$@"' sh \
  vllm serve "$MODEL_PATH" --host 0.0.0.0 --port "$port" \
  --served-model-name "${served_names[@]}" \
  "${parallel[@]}" --block-size "$block_size" --max-model-len "$max_model_len" \
  "${role_args[@]}" --enable-prefix-caching \
  --prefix-cache-retention-interval "$prefix_retention" \
  --kv-cache-dtype fp8 --speculative-config "$spec" \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  --kv-transfer-config "$moon" \
  --tool-call-parser glm47 --reasoning-parser glm47 \
  --enable-auto-tool-choice --trust-remote-code
