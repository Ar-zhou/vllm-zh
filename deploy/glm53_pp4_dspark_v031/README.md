# GLM-5.3 MG43 PP4 + DSpark on vLLM 0.31.0

This deployment runs on 192.168.31.35 with the vLLM 0.31.0 container image.
The public router is `http://192.168.31.35:8110`; Prefill is PP4 on GPUs
0–3 (`:8111`) with layer partition `22,20,20,16`, and Decode is DP4 on GPUs
4–7 (`:8112`). Both roles use DSpark K=8, FP8 KV, Mooncake RoCE, and LMCache.
The model name is `GLM-5.3-W4A4C8-MG43-BNT3-1P1D`; max context is 1,048,576.

Three source changes in this branch port the PP + DSpark fixes from
`Ar-zhou/vllm-zh` branch `deploy/glm52-pp8-dspark-v030` (commits `c9c91f77a02d`
and `bf50f0caab1d`) onto official vLLM tag `v0.31.0` (`db9527a46873`):

- Relay EAGLE/DSpark auxiliary hidden states across PP stages.
- Carry sparse MLA top-k indices across PP boundaries.
- Gather draft tokens on the main CUDA stream before broadcasting to PP ranks.

The fourth source change initializes LMCache's V3 cache-layout groups during
KV registration. The default V2 connector assumes a uniform cache shape and
fails on a long prompt because PP stages contain target MLA, DSA indexer, and
DSpark draft caches of different sizes. On PP rank 0, V2 allocated 23 pointers
but vLLM registered 29 caches. `lmcache.yaml` enables V3, and the source change
builds its groups before the first host allocation.

The image's LMCache V3 groups the heterogeneous caches, but its transfer path
still passes one global head size, block size, and stride to every group. The
`patches/lmcache_gpu_connectors.py` override passes each group's own shape and
KV format to the transfer kernel. This file is based on LMCache 0.5.5 as
shipped in the specified image; keep it in sync if the image is updated.
It also binds each DP worker's transfer streams to the device that owns its KV
tensors. `lmcache.yaml` disables LMCache's MLA default of saving only rank 0:
each PP stage owns different layers and must save its own KV data.

Prefill uses a fixed 48 GiB GPU KV allocation per stage and
`max-num-batched-tokens=8192`. The original auto-sized cache left too little
room for the sparse indexer's 36 GiB warmup workspace on the last PP rank.
The observed resulting P capacity is 1,704,832 tokens; D capacity is
1,570,816 tokens per replica.

The other `patches/` files are the compatibility overrides used by the previous
working 0.31.0 TP/DP deployment: DP4 MoE scratch allocation and CUDA device
selection, plus MultiConnector metrics handling. The DSpark draft checkpoint
is the existing MG43-configured draft at
`/mnt/paas/z00936200/glm53_mg43_pd_20261016/draft_mg43`.

## Install on 35

Copy this directory to
`/mnt/paas/z00936200/glm53_mg43_pp4_dspark_v031` on 35. Create its `source/`
directory and copy these files from this checkout into it:

| Repository path | Deployment path |
| --- | --- |
| `vllm/models/deepseek_v32/nvidia/model.py` | `source/model.py` |
| `vllm/model_executor/models/deepseek_v2.py` | `source/deepseek_v2.py` |
| `vllm/v1/worker/gpu/pp_utils.py` | `source/pp_utils.py` |
| `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py` | `source/lmcache_connector.py` |

The launcher bind-mounts these files and the `patches/` files over the matching
image paths. It does not modify the image. Copy the three unit files into `/etc/systemd/system/`,
run `systemctl daemon-reload`, and enable/start the P and D units. Start the
router after both backends pass `/health`.

```bash
systemctl status glm53-mg43-pp4-p glm53-mg43-pp4-d glm53-mg43-pp4-router
python3 /mnt/paas/z00936200/glm53_mg43_pp4_dspark_v031/smoke_check.py
python3 /mnt/paas/z00936200/glm53_mg43_pp4_dspark_v031/long_cache_check.py
journalctl -u glm53-mg43-pp4-p -u glm53-mg43-pp4-d -u glm53-mg43-pp4-router -f
```

`smoke_check.py` replays the short streaming request from `smoke_request.json`
twice. `long_cache_check.py` sends an ~11.5k-token prompt twice with
`temperature=0`; the second run has to succeed, which is what proves that every
PP stage stored and reloaded its own KV. Its generation budget defaults to
`max_tokens=1024` so the thinking output cannot consume the whole answer;
override it with `MAX_TOKENS=<n>`.

The previous TP deployment remains on 35 as disabled systemd units
`glm53-mg43-p`, `glm53-mg43-d`, and `glm53-mg43-router` and can be restarted
after stopping the PP units. It uses ports 8000, 8010, and 8020.
