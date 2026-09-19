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

These were measurements before the top-k handoff fix below. The eager configuration reached 99.3% of native TP8's accepted tokens per round, but the PP8 result was substantially slower. PP8 verifies seven rather than eight drafts, so its denominator is smaller.

## Memory balancing

The PP partition is `13,11,10,11,11,11,8,3`. It was chosen from per-layer safetensors byte counts plus measured PP-rank non-weight overhead, including the last-rank draft model and KV cache. KV memory budget remains 30G, with 377,280-token cache capacity and 1.89× concurrency at the configured 200,000-token max sequence length.

| Configuration | GPU 0–7 memory used (MiB) | Max–min |
|---|---|---:|
| Old `12,10,10,10,12,10,11,3`, Graph | 64681, 66385, 66545, 65865, 77525, 65645, 83673, 73221 | 18992 MiB |
| Balanced, Graph | 71193, 72967, 66907, 72949, 72967, 72963, 67051, 74843 | 7936 MiB |
| Balanced, eager | 72899, 74875, 68821, 74903, 74875, 74871, 68973, 77251 | 8430 MiB |

The eager peak was 77,251 MiB, versus 83,673 MiB on the old partition. The later top-k fix also repaired the Graph-mode acceptance regression; see below.

## Deployment and evidence

- Launcher: `/mnt/sfs_turbo/n30008093/glm52-dspark-native-v029-tp8/native_v029_pp8_run.sh` on 165. It includes `--enable-auto-tool-choice --tool-call-parser glm47`, port 8002, `--no-async-scheduling`, and `enable_adaptive_verification=false`. The later Graph-mode run removed `--enforce-eager`.
- Final benchmark: `/mnt/sfs_turbo/n30008093/glm52-dspark-v029-pp8/bench-short/acceptance-result.json`.
- Graph comparison: `bench-short-balanced-graph/acceptance-result.json` in the same directory.
- Target-only PP8 with the same partition and eager mode can also intermittently repeat `Comments` on a short chat prompt. This output variability is not specific to DSpark and is not resolved by the acceptance/memory fix. Raw completion smoke correctly continues `The capital of France is` with `Paris`.

## Earlier remaining-work list (superseded where noted below)

1. Diagnose the PP+DSpark CUDA Graph path; the balanced Graph run accepted only 1.711 draft tokens/round versus 2.039 in eager mode. **Resolved by the top-k handoff fix below.**
2. If throughput matters, schedule one extra target query row to verify all eight DSpark proposals instead of trimming to seven.
3. Investigate the intermittent target-only PP8 chat-output repetition separately from DSpark acceptance. **The top-k handoff fix below corrected the tested auto-tool request.**
4. Tool-call parsing needs separate work: an auto-tool request returned HTTP 200 but no structured `tool_calls` within 64 generated tokens; a forced `get_weather` choice returned HTTP 500 because xgrammar rejected GLM's `<tool_call>` token (154843). The auto-tool issue was corrected by the top-k handoff fix below; named tool verification remains separate.

## 2026-09-19 follow-up: PP target/tool-call isolation

- The checkpoint chat template recognizes `enable_thinking=false`, **not** `thinking=false`. The latter was used in an earlier smoke test and left template generation in thinking mode while the parser assumed otherwise. All comparisons below use `chat_template_kwargs={"enable_thinking": false}`.
- Native v0.29.0 **TP8 target-only** (no DSpark, eager), with the same model and `glm47` parser: three short France-capital chats all answered `Paris.`. Auto and named `get_weather` requests both returned HTTP 200 with a structured call and `{"city": "Beijing"}`.
- Patched v0.29.0 **PP8 target-only** (no DSpark, eager), same tool request: auto mode returned HTTP 200 but emitted malformed pseudo-tool tags (`<tool_call>get_weather` followed by `</invoke>` / `<parameter>`), with no `tool_calls`. Named mode returned HTTP 200 with `get_weather` arguments `{}` and `finish_reason=length`. Thus the tool failure is reproducible without DSpark and is upstream of tool parsing: the PP target generated a different token sequence.
- A diagnostic trace of the first 20 steps showed that PP0–PP7 had identical input token IDs, positions, and computed-token counts at every step of that request. The first generated token on PP8 was already different from TP8. A trial change making every synchronous PP consumer read the latest sampled-token broadcast did **not** improve tool output; it and all temporary instrumentation were reverted. The next investigation should compare target hidden states/logits and kernels across TP8 and PP8, starting at prefill, rather than changing DSpark or the `glm47` parser.
- Diagnostic logs are on 165 under `/mnt/sfs_turbo/n30008093/glm52-dspark-v029-pp8/`: `tp8-target-only-diagnostic.log`, `pp8-target-only-diagnostic.log`, `pp8-target-only-fixed.log` (the unsuccessful broadcast trial), and `pp8-input-diagnostic.log` (temporary numeric trace). The deployed PP8+DSpark launcher and code were restored after testing.

## 2026-09-19: Top-k index handoff across PP stages

The target-only PP failure came from the DeepSeek V3.2/GLM-5.2 top-k attention path. `DeepseekV32Model` kept `topk_indices_buffer` local to each PP stage; it was never included in `IntermediateTensors`. The checkpoint's `index_topk_freq=4` and `index_skip_topk_offset=3` mean only some layers recompute the indices. The balanced partition `13,11,10,11,11,11,8,3` starts several stages on layers that *reuse* the previous stage's indices. Those stages read stale/uninitialized local indices while all other input IDs and positions remain correct. The error can change the first generated token without throwing an exception.

As a diagnostic, a scorer-aligned partition `14,8,8,12,12,12,8,4` made the auto-tool call structured again, but its GPU memory usage was severely skewed (56.8–87.0 GiB). The code fix carries the int32 top-k indices in PP intermediate tensors, restores them on the next stage, and retains the custom intermediate-tensor factory after auxiliary-layer setup. The balanced partition is retained.

Same 12-request × 256-token benchmark, PP8, seven proposals/round, adaptive verification disabled:

| Balanced PP8 configuration | Accepted / proposed | Accepted draft tokens / round | Elapsed |
|---|---:|---:|---:|
| Before top-k fix, eager | 1554 / 5334 | 2.039 | 259.48 s |
| After top-k fix, eager | 1610 / 5138 | 2.193 | 250.47 s |
| Before top-k fix, CUDA Graph | 1417 / 5796 | 1.711 | 280.63 s |
| After top-k fix, CUDA Graph | 1705 / 4844 | 2.464 | 235.41 s |

The Graph run improves accepted draft tokens/round by 44% over the earlier Graph result and also beats patched eager. Its GPU 0–7 memory usage was 71,219, 73,077, 67,017, 73,057, 73,077, 73,073, 67,179, 74,937 MiB (7,920 MiB spread). The launch script now uses `FULL_DECODE_ONLY` CUDA Graph, with a ~74.9 GiB peak. Auto `get_weather` returned a structured `{"city":"Beijing"}` call. Native TP8's 2.055 accepted draft tokens/round is not a direct throughput comparison: PP8 still took 235 s versus TP8's 15.78 s on this small benchmark, and verifies one fewer proposal per round.

This fix does not by itself resolve forced/named tool choice. With strict tool calling, xgrammar returned HTTP 500 on the `<tool_call>` token; disabling strict mode avoided HTTP 500 but produced plain content instead of a structured tool call, so that is not a valid workaround. A request-scoped compatibility guard now discards DSpark drafts for structured-output requests, forcing one-token target verification under the grammar while leaving ordinary requests speculative. With `VLLM_ENFORCE_STRICT_TOOL_CALLING=True`, both auto and forced `get_weather` returned HTTP 200 with structured `{"city":"Beijing"}` arguments; a normal France-capital chat returned `Paris`. The 12×256 Graph acceptance benchmark above predates this guard, but its ordinary, non-structured requests do not enter that code path. The eighth draft and PP8 throughput remain open work.

## 2026-09-19: PP8 latency tuning

The final stable service remains synchronous (`--no-async-scheduling`). Enabling async scheduling consistently caused a device-side index assertion on PP7 in the sparse-attention path. `CUDA_LAUNCH_BLOCKING=1` masked the race and improved observed TPOT, but is a diagnostic setting rather than a production fix. Trials that cloned the PP top-k handoff, isolated target/draft top-k scratch buffers, changed request-finish ordering, or limited the async in-flight queue to eight batches did not make the non-blocking configuration stable; all trial source changes were reverted.

The retained tuning changes only `--max-num-batched-tokens` from 8192 to 16384. KV cache stays at 30G, `max-num-seqs` stays at 32, the balanced `13,11,10,11,11,11,8,3` partition is unchanged, and adaptive verification remains disabled.

Steady-state 16-request benchmark with unique prompts, about 4,354 input tokens and 128 output tokens per request:

| Prefill budget | Mean TTFT | Mean TPOT | Output throughput |
|---:|---:|---:|---:|
| 8192 | 19.59 s | 226.6 ms | 38.30 token/s |
| 16384 | 19.77 s | 197.0 ms | 41.47 token/s |

The 16384 configuration reduced TPOT by about 13% and increased output throughput by about 8%, while TTFT remained within 1%. An approximately 8,706-input-token run measured 38.42 s TTFT and 271.4 ms TPOT. After sustained tests, memory usage was 82,439, 85,683, 79,301, 86,173, 85,683, 85,485, 79,675, and 89,503 MiB on GPU 0–7 (10,202 MiB spread). Two consecutive slot-reuse tests completed and the service remained healthy. Named and auto tool calls both returned structured arguments successfully.

Evidence is under `/mnt/sfs_turbo/n30008093/glm52-dspark-v029-pp8/`: `perf-sync-base-comparable-unique.json`, `perf-sync-16384-comparable-warm-unique.json`, `perf-sync-16384-long-warm-unique.json`, `perf-sync-16384-stability-a.json`, `perf-sync-16384-stability-b.json`, `tool-named-16384.json`, and `tool-auto-16384.json`.
