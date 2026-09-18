# GLM-5.2 DSpark on vLLM 0.29.0: PP8 fix and validation

Branch: `glm52-dspark-pp8-v029` in `Ar-zhou/vllm-zh`. Tested on 192.168.1.165, API port 8002.

## Root cause and fix

The V2 PP handler originally consumed sampled output after an eight-step delay. That delay is appropriate for pipelined asynchronous scheduling, but this deployment uses synchronous PP scheduling. At the next DSpark verification step, PP0–PP6 therefore still had zero/stale draft tokens and sampled-token anchors, while PP7 had the current proposal. Target verification ran on different input IDs across stages, silently reducing acceptance.

The PP handler now broadcasts both sampled output and draft proposals and, for synchronous PP+DSpark only, consumes the latest completed broadcast before the next forward. Warmup entries are cleared. The port also carries globally indexed auxiliary hidden states across PP stages and bounds scheduled draft tokens by available query rows. The latter currently verifies seven of eight generated draft tokens per round; restoring the eighth needs a separate scheduler/query-row change.

## Measured acceptance

Fixed 12-request × 256-output-token synthetic benchmark, with the same target, draft checkpoint and prompts. `temperature=0`, adaptive verification disabled.

| Configuration | Accepted / proposed | Token acceptance | Accepted draft tokens / round | Elapsed |
|---|---:|---:|---:|---:|
| Native TP8, 8 drafts/round | 2071 / 8064 | 25.68% | 2.055 | 15.78 s |
| PP8 before synchronous-broadcast fix | 780 / 8064 | 9.67% | 0.774 | 388 s |
| PP8 old partition, eager, 7 drafts/round | 1538 / 5369 | 28.65% | 2.005 | 260.44 s |
| PP8 balanced partition, CUDA Graph, 7 drafts/round | 1417 / 5796 | 24.45% | 1.711 | 280.63 s |
| PP8 balanced partition, eager, 7 drafts/round | 1554 / 5334 | 29.13% | 2.039 | 259.48 s |

The final eager configuration reaches 99.3% of native TP8's accepted tokens per round. Since PP8 verifies seven rather than eight drafts, its 29.13% denominator is smaller; normalizing to eight proposals per round gives 25.49%, nearly TP8's 25.68%. This does not imply PP8 matches TP8 throughput: the PP8 run is much slower.

## Memory balancing

The PP partition is `13,11,10,11,11,11,8,3`. It was chosen from per-layer safetensors byte counts plus measured PP-rank non-weight overhead, including the last-rank draft model and KV cache. KV memory budget remains 30G, with 377,280-token cache capacity and 1.89× concurrency at the configured 200,000-token max sequence length.

| Configuration | GPU 0–7 memory used (MiB) | Max–min |
|---|---|---:|
| Old `12,10,10,10,12,10,11,3`, Graph | 64681, 66385, 66545, 65865, 77525, 65645, 83673, 73221 | 18992 MiB |
| Balanced, Graph | 71193, 72967, 66907, 72949, 72967, 72963, 67051, 74843 | 7936 MiB |
| Balanced, eager | 72899, 74875, 68821, 74903, 74875, 74871, 68973, 77251 | 8430 MiB |

The final eager peak is 77,251 MiB, versus 83,673 MiB on the old partition. Graph mode uses about 2–3 GiB less memory, but lowers acceptance with this PP+DSpark configuration; keep `--enforce-eager` until that path is repaired.

## Deployment and evidence

- Final launcher: `/mnt/sfs_turbo/n30008093/glm52-dspark-native-v029-tp8/native_v029_pp8_run.sh` on 165. It includes `--enable-auto-tool-choice --tool-call-parser glm47`, port 8002, `--no-async-scheduling`, `--enforce-eager`, and `enable_adaptive_verification=false`.
- Final benchmark: `/mnt/sfs_turbo/n30008093/glm52-dspark-v029-pp8/bench-short/acceptance-result.json`.
- Graph comparison: `bench-short-balanced-graph/acceptance-result.json` in the same directory.
- Target-only PP8 with the same partition and eager mode can also intermittently repeat `Comments` on a short chat prompt. This output variability is not specific to DSpark and is not resolved by the acceptance/memory fix. Raw completion smoke correctly continues `The capital of France is` with `Paris`.

## Remaining work

1. Diagnose the PP+DSpark CUDA Graph path; the balanced Graph run accepted only 1.711 draft tokens/round versus 2.039 in eager mode.
2. If throughput matters, schedule one extra target query row to verify all eight DSpark proposals instead of trimming to seven.
3. Investigate the intermittent target-only PP8 chat-output repetition separately from DSpark acceptance.
4. Tool-call parsing needs separate work: an auto-tool request returned HTTP 200 but no structured `tool_calls` within 64 generated tokens; a forced `get_weather` choice returned HTTP 500 because xgrammar rejected GLM's `<tool_call>` token (154843). The launcher has the tool flags, but this smoke test does not establish functional tool calling.
