# GLM-5.3 PP4 with dSpark sliding-window KV

This configuration keeps the PP layer partition at `22,20,20,16` and uses
four GPUs for one combined prefill/decode instance. The target KV uses
`fp8_ds_mla`; the dSpark draft KV uses `fp8`.
The tested target was
`/mnt/xds/sfs/GLM-5.3-NVFP4-W4A4C8-MG43-BNT3/v1`, with draft weights at
`/mnt/paas/lzs/glm5.3-dspark-lzs/sft-wtv-step800`.

Set these variables before starting vLLM:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_PP_LAYER_PARTITION=22,20,20,16
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_KV_CACHE_LAYOUT=BLHNC
export VLLM_SWA_PREFILL_INFLIGHT_BATCHES=1
```

Use `--pipeline-parallel-size 4 --tensor-parallel-size 1`,
`--block-size 64`, `--max-num-batched-tokens 32768`,
`--max-num-seqs 8`, `--max-model-len 200000`, and
`--gpu-memory-utilization 0.88`. Set the speculative method to `dspark`,
`num_speculative_tokens` to `8`, `draft_tensor_parallel_size` to `1`,
and `kv_cache_dtype` to `fp8`. On the tested GPU, utilization `0.92`
left too little room for a 150k-token prefill's temporary allocation.

`BLHNC` keeps the four sliding-window draft layers in their own KV group,
so old blocks can be recycled. `VLLM_SWA_PREFILL_INFLIGHT_BATCHES=1`
limits each request to one unfinished prefill batch at a time; the KV
admission calculation uses the same limit. This reduces the reserved SWA
window but can increase the latency of a single long prefill. Multiple
requests can still fill the PP pipeline.

On a four-GPU NVIDIA H800 host, this setup reported a 4,138,599-token
group-aware KV capacity (20.69 concurrent 200k-token requests). A
150,004-token request completed without an OOM or draft-verification
assertion. Capacity depends on the model weights, hardware, and available
GPU memory.
