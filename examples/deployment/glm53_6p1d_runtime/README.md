# 6P1D runtime snapshot (vLLM 0.31.0)

This branch records the source files used for the 6-prefill / 1-decode
benchmark on 2026-10-26. It is based on the PP4 SWA/DSpark P/D branch at
`65bad2204be2c848652e6f2c1c888239212ed2f2`.

The vLLM files are placed at their import paths in this branch. The files in
`lmcache_overrides/` are byte-for-byte snapshots of files bind-mounted over
the LMCache package in the vLLM 0.31.0 image; they are **not** imported by the
vLLM source tree. The two files in `vllm_overrides/` were also bind-mounted
into vLLM at the destinations below. Source files were compared across the
prefill and decode hosts and had matching SHA-256 hashes.

| Role | Runtime configuration |
| --- | --- |
| Prefill | Six replicas, each TP1/PP2, 40/38 target layers, FP8 KV, block size 64 |
| Decode | One DP4/EP4 replica, TP1 per worker, FP8 KV, block size 64 |
| Both | 1,048,576 max model tokens, DSpark K=7, probabilistic draft, block rejection, adaptive verification off |
| Prefill scheduler | 8,192 max batched tokens, 8 max sequences, 4,096 long-prefill threshold, GPU prefix caching on |
| Decode scheduler | 256 max batched tokens, 32 max sequences, GPU prefix caching on |
| LMCache | Prefill only; 2,200 GiB L1 per host, 2,000 GiB initially pinned, LRU, 1,024-token chunks, P2P enabled |
| Router | Consistent-hash P/D routing; Mooncake RDMA KV transfer |

All serving instances used `enable_flashinfer_autotune=false`. The prefill
side used `VLLM_MAX_INFLIGHT_ASYNC_KV_LOADS=64` and
`VLLM_P_ASYNC_LOAD_BUDGET_BYPASS=1`; decode used the same load cap set to 16.

## Files to inspect

- [`vllm/v1/core/sched/scheduler.py`](../../../vllm/v1/core/sched/scheduler.py):
  bounded SWA prefix replay, async LMCache admissions and queue diagnostics.
- [`vllm/v1/core/kv_cache_utils.py`](../../../vllm/v1/core/kv_cache_utils.py):
  PP/SWA KV group sizing.
- [`vllm/model_executor/models/dspark_context_kv.py`](../../../vllm/model_executor/models/dspark_context_kv.py):
  context-only prefill draft KV.
- [`vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py`](../../../vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py):
  vLLM to LMCache MP connector behavior.
- [`lmcache_overrides/`](lmcache_overrides/): LMCache package overrides,
  including P2P ranked lookup/prefetch and pinned-memory allocation.

## LMCache override destinations

| File in this directory | Package path within the image |
| --- | --- |
| `lmcache_overrides/lmcache_mp_connector.py` | `lmcache/integration/vllm/lmcache_mp_connector.py` |
| `lmcache_overrides/vllm_multi_process_adapter.py` | `lmcache/integration/vllm/vllm_multi_process_adapter.py` |
| `lmcache_overrides/gpu_connectors.py` | `lmcache/v1/gpu_connector/gpu_connectors.py` |
| `lmcache_overrides/lazy_memory_allocator.py` | `lmcache/v1/memory_allocators/lazy_memory_allocator.py` |
| `lmcache_overrides/engine_context.py` | `lmcache/v1/multiprocess/engine_context.py` |
| `lmcache_overrides/lookup.py` | `lmcache/v1/multiprocess/modules/lookup.py` |
| `lmcache_overrides/lmcache_driven_transfer.py` | `lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py` |
| `lmcache_overrides/prefetch_controller.py` | `lmcache/v1/distributed/storage_controllers/prefetch_controller.py` |
| `lmcache_overrides/api.py` | `lmcache/v1/distributed/api.py` |
| `vllm_overrides/cutlass_moe.py` | `vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py` |
| `vllm_overrides/worker_utils.py` | `vllm/v1/worker/utils.py` |

The machine-specific launch scripts and model paths are intentionally omitted
from this public snapshot. The table above records the effective performance
parameters; the source files and overrides are the exact bytes used in the
benchmark.
