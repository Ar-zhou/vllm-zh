# PP4 + DSpark 1P1D launcher (vLLM 0.31.0)

This branch carries the PP auxiliary-state, sparse-index, and draft-token
broadcast fixes from the PP adaptation. It does **not** include the later
sliding-window KV group-splitting or scheduler patches. The launcher uses the
vLLM 0.31.0 image plus read-only bind mounts for the three PP source files and
two DP compatibility files in this checkout.

The launcher starts one PP4 prefill engine, one DP4 decode engine, and a host
`vllm-router` process. It uses DSpark K=8 on both engines, FP8 KV, Mooncake
RDMA, and a 1,048,576-token context. LMCache is not enabled. No service is
started by checking out this branch.

## Required environment

Set these variables in a private file outside this repository or pass them to
the command. Do not commit actual values:

| Variable | Meaning |
| --- | --- |
| `MODEL_PATH` | Absolute path to the target model directory |
| `DRAFT_MODEL_PATH` | Absolute path to the DSpark draft directory |
| `SERVED_MODEL_NAME` | Public API model name |
| `MOONCAKE_DEVICE_NAME` | Comma-separated RDMA device list |

Example with placeholders:

```bash
export MODEL_PATH=/models/target
export DRAFT_MODEL_PATH=/models/draft
export SERVED_MODEL_NAME=my-model
export MOONCAKE_DEVICE_NAME=mlx5_0
export CONTAINERD_NAMESPACE=default

./deploy/pp_dspark_1p1d_no_swa_kv/run.sh p
./deploy/pp_dspark_1p1d_no_swa_kv/run.sh d
./deploy/pp_dspark_1p1d_no_swa_kv/run.sh router
```

Run the three commands in separate supervisors or terminals. Start the router
after the P and D `/health` endpoints respond. The default ports are 8111 (P),
8112 (D), and 8110 (router); the Mooncake bootstrap port is 8998. Set
`VLLM_ROUTER_BIN` if `vllm-router` is not on `PATH`.
If either checkpoint directory contains symlinks to weight files outside that
directory, the launcher automatically bind-mounts the resolved target
directories into the container. The real paths stay in the private host
environment and are never written to this repository.

The defaults target eight GPUs: P uses GPUs 0–3, PP4 with partition
`22,20,20,16`, 8192 batched tokens, eight sequences, and a 76 GiB KV pool per
stage. D uses GPUs 4–7, DP4, 128 batched tokens, and 32 sequences. The 76 GiB
P pool was used by the pre-SWA-optimization deployment to leave room for the
sparse indexer's warmup workspace. Tune it for the target hardware through
`P_KV_CACHE_MEMORY_BYTES` before increasing P's batch limit.

The script enables GPU Prefix Cache with
`PREFIX_CACHE_RETENTION_INTERVAL=0` by default, matching the original PP
deployment. With the draft's sliding-window cache, this may miss a shorter
request following a longer request with the same prefix. Set
`PREFIX_CACHE_RETENTION_INTERVAL=4096` for periodic checkpoints if that reuse
matters. This is a cache-retention policy; it does not apply the later SWA KV
group-width optimization. The setting trades retained KV space against
recomputed tail tokens.

Other optional variables include `VLLM_IMAGE`, `P_GPU_IDS`, `D_GPU_IDS`,
`PREFILL_PORT`, `DECODE_PORT`, `ROUTER_PORT`, `MOONCAKE_BOOTSTRAP_PORT`,
`P_MAX_NUM_BATCHED_TOKENS`, `D_MAX_NUM_BATCHED_TOKENS`, `P_MAX_NUM_SEQS`,
`D_MAX_NUM_SEQS`, `P_GPU_MEMORY_UTILIZATION`, `D_GPU_MEMORY_UTILIZATION`,
`MAX_MODEL_LEN`, `DSPARK_TOKENS`, `PP_LAYER_PARTITION`, and `MODEL_ALIAS`.

This is a launcher and source snapshot. It has been checked for shell syntax
and argument construction; starting the model requires the specified weights,
RDMA devices, GPUs, and router on the deployment host.
