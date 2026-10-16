#!/usr/bin/env bash
set -Eeuo pipefail
role=${1:?p or d}
base=/mnt/paas/z00936200/glm53_mg43_pp4_dspark_v031
image=m.daocloud.io/docker.io/vllm/vllm-openai:v0.31.0
model=/mnt/xds/sfs/GLM-5.3-NVFP4-W4A4C8-MG43-BNT3/v1
draft=/mnt/paas/z00936200/glm53_mg43_pd_20261016/draft_mg43
name="glm53-mg43-pp4-${role}"

for path in "$model/config.json" "$draft/config.json" "$base/lmcache.yaml" \
  "$base/source/model.py" "$base/source/deepseek_v2.py" "$base/source/pp_utils.py" \
  "$base/source/lmcache_connector.py"; do
  [[ -r "$path" ]] || { echo "Missing $path" >&2; exit 1; }
done

spec='{"method":"dspark","model":"'"$draft"'","num_speculative_tokens":8,"draft_tensor_parallel_size":1,"draft_sample_method":"greedy","rejection_sample_method":"block","enable_adaptive_verification":false,"attention_backend":"FLASH_ATTN","kv_cache_dtype":"fp8"}'
moon='{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"mooncake_protocol":"rdma","device_name":"mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7,mlx5_bond_8"}}'
patch_mounts=(
  --mount "type=bind,src=$base/source/model.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v32/nvidia/model.py,options=rbind:ro"
  --mount "type=bind,src=$base/source/deepseek_v2.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py,options=rbind:ro"
  --mount "type=bind,src=$base/source/pp_utils.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/pp_utils.py,options=rbind:ro"
  --mount "type=bind,src=$base/source/lmcache_connector.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py,options=rbind:ro"
  --mount "type=bind,src=$base/patches/lmcache_gpu_connectors.py,dst=/usr/local/lib/python3.12/dist-packages/lmcache/v1/gpu_connector/gpu_connectors.py,options=rbind:ro"
  --mount "type=bind,src=$base/patches/multi_connector_metricsfix.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py,options=rbind:ro"
)

if [[ "$role" == p ]]; then
  port=8111; gpus=0,1,2,3
  parallel=(--tensor-parallel-size 1 --pipeline-parallel-size 4 --distributed-executor-backend mp)
  role_args=(--gpu-memory-utilization 0.90 --kv-cache-memory-bytes 48G --max-num-batched-tokens 8192 --safetensors-load-strategy lazy)
  envs=(--env VLLM_PP_LAYER_PARTITION=22,20,20,16 --env VLLM_MOONCAKE_BOOTSTRAP_PORT=8999)
  kv='{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_connector_extra_config":{"connectors":['"$moon"',{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}]}}'
elif [[ "$role" == d ]]; then
  port=8112; gpus=4,5,6,7
  parallel=(--tensor-parallel-size 1 --data-parallel-size 4 --data-parallel-backend mp --data-parallel-size-local 4 --enable-expert-parallel --distributed-executor-backend mp)
  role_args=(--max-num-seqs 32 --max-num-batched-tokens 1024 --gpu-memory-utilization 0.94 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}')
  patch_mounts+=(
    --mount "type=bind,src=$base/patches/cutlass_moe_dpfix.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py,options=rbind:ro"
    --mount "type=bind,src=$base/patches/worker_utils_dpfix.py,dst=/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/utils.py,options=rbind:ro"
  )
  envs=()
  moon=${moon/kv_producer/kv_consumer}
  kv='{"kv_connector":"MultiConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"connectors":['"$moon"',{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}]}}'
else
  exit 2
fi

IFS=, read -r -a gpu_ids <<< "$gpus"
gpu_flags=()
for gpu in "${gpu_ids[@]}"; do gpu_flags+=(--gpus "$gpu"); done

exec ctr -n dspark-v41 run --rm --net-host "${gpu_flags[@]}" --device /dev/infiniband \
  --cap-add CAP_SYS_RESOURCE --cap-add CAP_IPC_LOCK \
  "${patch_mounts[@]}" \
  --mount "type=bind,src=/dev/shm,dst=/dev/shm,options=rbind:rw" \
  --mount "type=bind,src=$model,dst=$model,options=rbind:ro" \
  --mount "type=bind,src=/opt/glm53-perf-weights,dst=/opt/glm53-perf-weights,options=rbind:ro" \
  --mount "type=bind,src=/mnt/paas/z00936200/glm53_mg43_pd_20261016,dst=/mnt/paas/z00936200/glm53_mg43_pd_20261016,options=rbind:ro" \
  --mount "type=bind,src=$base,dst=$base,options=rbind:rw" \
  --env VLLM_ENGINE_READY_TIMEOUT_S=3600 --env VLLM_HTTP_TIMEOUT_KEEP_ALIVE=75 \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env PYTHONHASHSEED=0 \
  --env "LMCACHE_CONFIG_FILE=$base/lmcache.yaml" \
  "${envs[@]}" "$image" "$name" \
  sh -c 'ulimit -l unlimited; ulimit -n 1048576; exec "$@"' sh \
  vllm serve "$model" --host 0.0.0.0 --port "$port" \
  --served-model-name GLM-5.3-W4A4C8-MG43-BNT3-1P1D \
  "${parallel[@]}" --block-size 64 --max-model-len 1048576 \
  "${role_args[@]}" --enable-prefix-caching --kv-cache-dtype fp8 \
  --speculative-config "$spec" \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  --kv-transfer-config "$kv" \
  --tool-call-parser glm47 --reasoning-parser glm47 --enable-auto-tool-choice \
  --trust-remote-code
