#!/usr/bin/env bash
set -Eeuo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the target checkpoint}"
: "${DRAFT_PATH:?Set DRAFT_PATH to the MLA draft checkpoint}"
mode=${1:?baseline, shared_first, or first_layer}
port=${2:-8000}
case "$mode" in
  baseline|shared_first|first_layer) ;;
  *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac

export VLLM_DSPARK_MLA_KV_ABLATION="$mode"
export VLLM_KV_CACHE_LAYOUT=BLHNC
export VLLM_USE_V2_MODEL_RUNNER=1

spec=$(printf '{"method":"dspark","model":"%s","num_speculative_tokens":8,"draft_tensor_parallel_size":1,"draft_sample_method":"probabilistic","rejection_sample_method":"standard","enable_adaptive_verification":false,"attention_backend":"FLASH_ATTN","kv_cache_dtype":"bfloat16"}' "$DRAFT_PATH")

exec vllm serve "$MODEL_PATH" --host 0.0.0.0 --port "$port" \
  --served-model-name glm53-dspark-one-kv \
  --tensor-parallel-size 1 --data-parallel-size 4 \
  --data-parallel-size-local 4 --data-parallel-backend mp \
  --enable-expert-parallel --distributed-executor-backend mp \
  --block-size 64 --max-model-len 1048576 \
  --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.88 --safetensors-load-strategy lazy \
  --enforce-eager --no-enable-prefix-caching --kv-cache-dtype fp8 \
  --speculative-config "$spec" \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  --tool-call-parser glm47 --reasoning-parser glm47 --enable-auto-tool-choice \
  --trust-remote-code
