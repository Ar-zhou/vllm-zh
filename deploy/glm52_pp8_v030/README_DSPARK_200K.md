# GLM-5.2, vLLM 0.30.0, PP8 + DSpark, 200k context

On `the deployment host` (8 × H800), use the source checkout at
`<checkout>` and run:

```bash
bash deploy/glm52_pp8_v030/prepare_runtime.sh
tmux new-session -d -s glm52-v030-pp8-dspark-200k \
  'cd <checkout> && bash deploy/glm52_pp8_v030/start_dspark_bf16_200k.sh'
```

This uses the GLM-5.2 NVFP4 MG39 target and `step400` DSpark draft,
TP1 × PP8, speculative k=7, no adaptive verification, 200,000-token maximum
length, 30 GiB BF16 KV per rank, target `FLASHMLA_SPARSE`, draft
`FLASH_ATTN`, and the `glm47` tool-call parser. The served name is
`glm-5.2-dspark` on port 8002. Logs are under
`$repo_root/.runtime/logs/`.

vLLM 0.30 uses the new `vllm/models/deepseek_v32/nvidia/model.py` path for
this target. The branch adds PP relay of DSpark auxiliary hidden states using
the 0.30 PP interface. It also keeps the draft KV as BF16; its
`FLASH_ATTN` backend rejected FP8 KV during the separate 1M experiment.

Verified on 2026-09-24: `/health` returned HTTP 200, `/v1/models` reported
`glm-5.2-dspark` with `max_model_len=200000`, the KV planner reported
348,032-token capacity, and a short chat request generated a response.
These checks do not establish acceptance rate, throughput, or long-context
quality. Attach with `tmux attach -t glm52-v030-pp8-dspark-200k` and stop with
Ctrl-C in that session.
