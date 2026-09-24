#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data_root="${GLM52_V030_DATA_ROOT:-$repo_root/.runtime}"
python_bin="${GLM52_V030_PYTHON:-python}"
: "${GLM52_V030_MODEL:?Set GLM52_V030_MODEL to the target checkpoint}"
model_path="$GLM52_V030_MODEL"
log_dir="$data_root/logs"
mkdir -p "$log_dir"
if [[ ! -f "$repo_root/vllm/_version.py" || ! -f "$repo_root/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so" ]]; then
  echo "vLLM 0.30 runtime is missing; run prepare_runtime.sh first" >&2
  exit 1
fi

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
unset VLLM_ROOT
export PATH="$(dirname "$python_bin"):${CUDA_HOME:+$CUDA_HOME/bin:}:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME:+$CUDA_HOME/lib64:$CUDA_HOME/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="${CUDA_HOME:-}"
export PYTHONUNBUFFERED=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENFORCE_STRICT_TOOL_CALLING=True
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_NVLS_ENABLE=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0

log_path="$log_dir/server-foreground-glm52-pp8-mtp3-v030-$(date +%Y%m%d-%H%M%S).log"
echo "[startup] log: $log_path"
cd "$repo_root"
"$python_bin" -m vllm.entrypoints.openai.api_server \
  --model "$model_path" \
  --served-model-name glm-5.2-mtp \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --max-model-len 200000 \
  --kv-cache-dtype bfloat16 \
  --kv-cache-memory-bytes 12G \
  --block-size 64 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 16384 \
  --attention-backend FLASHMLA_SPARSE \
  --safetensors-load-strategy lazy \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --host 0.0.0.0 \
  --port 8002 2>&1 | tee "$log_path"
