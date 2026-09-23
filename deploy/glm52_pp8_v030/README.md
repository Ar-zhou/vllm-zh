# GLM-5.2 on vLLM 0.30.0, PP8

This is the 192.168.1.165 (8 × H800) deployment experiment. It uses the
`releases/v0.30.0` branch of `Ar-zhou/vllm-zh` as its source base. The model is
GLM-5.2 NVFP4 with built-in MTP k=3, **not DSpark**. The target is TP1 × PP8,
200k max context, BF16 KV, and port 8002.
`--safetensors-load-strategy lazy` avoids vLLM 0.30's automatic NFS
checkpoint prefetch, which stalled the first launch under concurrent PP loads.
The initial 30 GiB-per-rank KV budget yielded 2.70M token capacity but caused
rank 7 to OOM during warmup; the script uses 12 GiB for the 200k smoke test.

Run from this checkout:

```bash
bash deploy/glm52_pp8_v030/prepare_runtime.sh
bash deploy/glm52_pp8_v030/start.sh
```

For a service that survives SSH disconnects while retaining live logs:

```bash
tmux new-session -d -s glm52-v030-pp8 \
  'cd /mnt/mtp/sfs/z00936200/vllm-zh-v030-glm52-pp8 && bash deploy/glm52_pp8_v030/start.sh'
tmux attach -t glm52-v030-pp8
```

`prepare_runtime.sh` fetches the official vLLM 0.30.0 CUDA 13 wheel from the
configured PyPI mirror and unpacks its compiled extensions and bundled
third-party runtime into this source checkout. These generated files are
ignored by Git. The Python dependency environment is the existing CUDA 13
environment at `/mnt/sfs_turbo/n30008093/vllm/.venv`; matching vLLM 0.30.0
source and extensions are used via `PYTHONPATH`. It does not overwrite the
existing 0.29 installation.

Logs and wheel cache live under
`/mnt/mtp/sfs/z00936200/glm52-pp8-v030/`. The service runs in the foreground.
Stop it with Ctrl-C. Validate with `curl -fsS http://127.0.0.1:8002/health`
and a chat or completion request against `glm-5.2-mtp`.

The vLLM 0.30 startup warning says k>1 repeatedly executes one MTP layer and
may lower acceptance. A successful short request is only a deployment smoke
test, not a correctness, acceptance-rate, or long-context benchmark.

## Verified on 2026-09-23

- Server: `/health` returned HTTP 200; `/v1/models` reported
  `glm-5.2-mtp` with `max_model_len=200000`.
- PP partition: `[9,10,10,10,10,10,10,9]` across GPUs 0–7.
- 12 GiB KV cache: 1,081,280-token reported capacity, about 5.41 full-length
  200k requests in the cache estimate. This is not a concurrency benchmark.
- One 32-token chat request and one 16-token completion both returned HTTP
  200. The completion began with `2` for `1+1=`. The chat response was
  truncated in its reasoning text, so no accuracy claim is made.
- After those two requests, service-wide MTP counters were 48 draft tokens,
  32 accepted tokens, with accepted counts `[14,10,8]` by draft position.
  This tiny sample is not an acceptance-rate benchmark.
- GPU memory used (MiB): `[58636,77963,77871,77963,77871,77963,77871,119815]`.
  The final stage is still much less memory-balanced than the others.
- Live session: `tmux attach -t glm52-v030-pp8`; log:
  `/mnt/mtp/sfs/z00936200/glm52-pp8-v030/logs/server-foreground-glm52-pp8-mtp3-v030-20260923-230243.log`.
