#!/usr/bin/env bash
set -Eeuo pipefail
# Both P and D use the same approximate four-cache layout. Layer 5 still
# executes, but reads layer 4's KV and does not register/write a fifth cache.
# Keeping the layouts identical makes Mooncake/LMCache metadata compatible.
role=${1:?p or d}
base=/mnt/paas/z00936200/glm53_pd_swa_lmcache_v031
mla_base=/mnt/paas/z00936200/glm53_mla_pd_v031
image=m.daocloud.io/docker.io/vllm/vllm-openai:v0.31.0
model=/mnt/xds/sfs/GLM-5.3-NVFP4-W4A4C8-MG43-BNT3/v1
draft=/mnt/paas/lzs/glm5.3-mla-dspark-lzs/4swa_1fa/mla-sft-step2400
name="glm53-mla-pd-${role}"

for path in "$model/config.json" "$draft/config.json" "$base/lmcache.yaml" "$mla_base/source/glm_dspark_mla.py" \
  "$base/source/model.py" "$base/source/deepseek_v2.py" "$base/source/pp_utils.py" \
  "$base/source/lmcache_connector.py" "$base/source/lmcache_mp_connector.py" "$base/source/vllm_multi_process_adapter.py" "$base/source/external_lmcache_mp_connector.py" "$base/source/vllm.py" \
  "$base/source/kv_cache_utils.py" "$base/source/scheduler.py"; do
  [[ -r "$path" ]] || { echo "Missing $path" >&2; exit 1; }
done

spec='{"method":"dspark","model":"'"$draft"'","num_speculative_tokens":8,"draft_tensor_parallel_size":1,"draft_sample_method":"probabilistic","rejection_sample_method":"standard","enable_adaptive_verification":false,"attention_backend":"FLASH_ATTN","kv_cache_dtype":"bfloat16"}'
spec_args=(--speculative-config "$spec")
if [[ "${GLM_DISABLE_DSPARK:-0}" == 1 ]]; then spec_args=(); fi
moon='{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"mooncake_protocol":"rdma","device_name":"mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7,mlx5_bond_8"}}'
patch_mounts=(
  --mount "type=bind,src=$mla_base/source/speculative.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py,options=rbind:ro"
  --mount "type=bind,src=$mla_base/source/glm_dspark_mla.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/glm_dspark_mla.py,options=rbind:ro"
  --mount "type=bind,src=$mla_base/source/qwen3_dspark.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dspark.py,options=rbind:ro"
  --mount "type=bind,src=$mla_base/source/registry.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/registry.py,options=rbind:ro"
  --mount "type=bind,src=$mla_base/source/algos.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/configs/speculators/algos.py,options=rbind:ro"
  --mount "type=bind,src=$mla_base/source/model_arch_config_convertor.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/model_arch_config_convertor.py,options=rbind:ro"
  --mount "type=bind,src=$mla_base/source/dspark_mla_backend.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/dspark.py,options=rbind:ro"
  --mount "type=bind,src=$base/source/vllm.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/config/vllm.py,options=rbind:ro"
  --mount "type=bind,src=$base/source/kv_cache_utils.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py,options=rbind:ro"
  --mount "type=bind,src=$base/source/scheduler.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py,options=rbind:ro"
  --mount "type=bind,src=$base/source/model.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v32/nvidia/model.py,options=rbind:ro"
  --mount "type=bind,src=$base/source/deepseek_v2.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py,options=rbind:ro"
  --mount "type=bind,src=$base/source/pp_utils.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/pp_utils.py,options=rbind:ro"
  --mount "type=bind,src=$base/source/lmcache_connector.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py,options=rbind:ro"
  --mount "type=bind,src=$base/source/lmcache_mp_connector.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py,options=rbind:ro"
  --mount "type=bind,src=$base/source/external_lmcache_mp_connector.py,dst=/usr/local/lib/python3.12/dist-packages/lmcache/integration/vllm/lmcache_mp_connector.py,options=rbind:ro"
  --mount "type=bind,src=$base/source/vllm_multi_process_adapter.py,dst=/usr/local/lib/python3.12/dist-packages/lmcache/integration/vllm/vllm_multi_process_adapter.py,options=rbind:ro"
  --mount "type=bind,src=$base/patches/lmcache_gpu_connectors.py,dst=/usr/local/lib/python3.12/dist-packages/lmcache/v1/gpu_connector/gpu_connectors.py,options=rbind:ro"
  --mount "type=bind,src=$base/patches/multi_connector_metricsfix.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py,options=rbind:ro"
)

if [[ "$role" == p ]]; then
  port=8111; gpus=0,1,2,3
  flashinfer_autotune=false
  lmcache_mp_port=5556
  lmcache_api_port_start=6999
  parallel=(--tensor-parallel-size 1 --pipeline-parallel-size 4 --distributed-executor-backend mp)
  role_args=(--gpu-memory-utilization 0.90 --prefix-cache-retention-interval 0 --max-num-batched-tokens "${GLM_P_MAX_BATCH_TOKENS:-8192}" --max-num-seqs "${GLM_P_MAX_NUM_SEQS:-8}" --safetensors-load-strategy lazy --enforce-eager)
  if [[ "${GLM_P_LONG_PREFILL_THRESHOLD:-0}" != 0 ]]; then
    role_args+=(--long-prefill-token-threshold "$GLM_P_LONG_PREFILL_THRESHOLD")
  fi
  envs=(--env VLLM_DSPARK_MLA_KV_ABLATION=shared_first_4 --env VLLM_PP_LAYER_PARTITION=22,20,20,16 --env VLLM_MOONCAKE_BOOTSTRAP_PORT=8999 --env VLLM_KV_CACHE_LAYOUT=BLHNC --env VLLM_MAX_INFLIGHT_ASYNC_KV_LOADS=64 --env VLLM_P_QUEUE_DIAG=1 --env VLLM_P_ASYNC_LOAD_BUDGET_BYPASS=1)
  if [[ "${GLM_MOONCAKE_DIAGNOSTICS:-0}" == 1 ]]; then
    [[ -r "$base/source/mooncake_connector.py" ]] || {
      echo "Missing $base/source/mooncake_connector.py" >&2; exit 1;
    }
    patch_mounts+=(
      --mount "type=bind,src=$base/source/mooncake_connector.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py,options=rbind:ro"
    )
    envs+=(--env VLLM_MOONCAKE_SENDER_DIAGNOSTICS=1)
  fi
elif [[ "$role" == d ]]; then
  port=8112; gpus=4,5,6,7
  flashinfer_autotune=false
  parallel=(--tensor-parallel-size 1 --data-parallel-size 4 --data-parallel-backend mp --data-parallel-size-local 4 --enable-expert-parallel --distributed-executor-backend mp)
  role_args=(--max-num-seqs 32 --max-num-batched-tokens 256 --gpu-memory-utilization 0.94 --enforce-eager)
  patch_mounts+=(
    --mount "type=bind,src=$base/patches/cutlass_moe_dpfix.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py,options=rbind:ro"
    --mount "type=bind,src=$base/patches/worker_utils_dpfix.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/utils.py,options=rbind:ro"
  )
  envs=(--env VLLM_DSPARK_MLA_KV_ABLATION=shared_first_4 --env VLLM_KV_CACHE_LAYOUT=BLHNC --env VLLM_SWA_PD_GROUP_SPLIT=3 --env VLLM_MAX_INFLIGHT_ASYNC_KV_LOADS=16)
  moon=${moon/kv_producer/kv_consumer}
else
  exit 2
fi

# Mooncake handles P/D transfer. Only P uses LMCache MP for prefix KV.
kv=$moon
if [[ "$role" == p ]]; then
  lmcache_mp='{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://127.0.0.1","lmcache.mp.port":'"$lmcache_mp_port"',"lmcache.mp.eager_prefetch":true}}'
  kv='{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_connector_extra_config":{"connectors":['"$moon"','"$lmcache_mp"']}}'
  envs+=(
    --env LMCACHE_INTERNAL_API_SERVER_ENABLED=true
    --env LMCACHE_INTERNAL_API_SERVER_HOST=127.0.0.1
    --env "LMCACHE_INTERNAL_API_SERVER_PORT_START=$lmcache_api_port_start"
    --env "LMCACHE_CONFIG_FILE=$base/lmcache.yaml"
  )
fi

IFS=, read -r -a gpu_ids <<< "$gpus"
gpu_flags=()
for gpu in "${gpu_ids[@]}"; do gpu_flags+=(--gpus "$gpu"); done

exec ctr -n dspark-v41 run --rm --net-host "${gpu_flags[@]}" --device /dev/infiniband \
  --cap-add CAP_SYS_RESOURCE --cap-add CAP_IPC_LOCK \
  "${patch_mounts[@]}" \
  --mount "type=bind,src=/dev/shm,dst=/dev/shm,options=rbind:rw" \
  --mount "type=bind,src=$model,dst=$model,options=rbind:ro" \
  --mount "type=bind,src=$draft,dst=$draft,options=rbind:ro" \
  --mount "type=bind,src=/opt/glm53-perf-weights,dst=/opt/glm53-perf-weights,options=rbind:ro" \
  --mount "type=bind,src=/mnt/paas/z00936200/glm53_mg43_pd_20261016,dst=/mnt/paas/z00936200/glm53_mg43_pd_20261016,options=rbind:ro" \
  --mount "type=bind,src=$base,dst=$base,options=rbind:rw" \
  --mount "type=bind,src=$mla_base,dst=$mla_base,options=rbind:ro" \
  --env VLLM_ENGINE_READY_TIMEOUT_S=3600 --env VLLM_HTTP_TIMEOUT_KEEP_ALIVE=75 \
  --env VLLM_SERVER_DEV_MODE=1 \
  --env VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=600 \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env PYTHONHASHSEED=0 \
  "${envs[@]}" "$image" "$name" \
  sh -c 'ulimit -l unlimited; ulimit -n 1048576; exec "$@"' sh \
  vllm serve "$model" --host 0.0.0.0 --port "$port" \
  --served-model-name GLM-5.3-MLA-DSpark-PP4-1P1D glm-5.3-dspark \
  "${parallel[@]}" --block-size 64 --max-model-len 1048576 \
  "${role_args[@]}" --enable-prefix-caching --kv-cache-dtype fp8 \
  "${spec_args[@]}" \
  --kernel-config "{\"enable_flashinfer_autotune\":$flashinfer_autotune}" \
  --kv-transfer-config "$kv" \
  --tool-call-parser glm47 --reasoning-parser glm47 --enable-auto-tool-choice \
  --trust-remote-code
