# dSpark acceptance gap: PP4 prefill + TP4 decode versus colocated TP8

This note records a vLLM 0.29.0 experiment for follow-up on another eight-GPU
machine. It contains aggregate counters and generic configuration only. It
does not contain request payloads, hostnames, network addresses, user names,
GPU identifiers, or checkpoint locations.

## What was observed

Both runs used the same target model family and dSpark checkpoint, eight
speculative tokens, draft TP1, prefix caching, asynchronous decode scheduling,
and disabled confidence-based adaptive verification. The disaggregated run
used one TP1/PP4 prefill engine and one TP4 decode engine, linked by NIXL push.
The colocated run used one TP8 engine for both prefill and decode.

| Decode-side metric | PP4 prefill + TP4 decode | Colocated TP8 |
| --- | ---: | ---: |
| Draft rounds | 28,120 | 36,178 |
| Proposed draft tokens | 224,960 | 289,424 |
| Accepted draft tokens | 30,594 | 113,613 |
| Token acceptance | 13.5998% | 39.2549% |
| Mean accepted length / round | 1.0880 | 3.1404 |
| Completed requests | 94 | 240 |
| Prompt tokens across completed requests | 4,523,219 | 8,048,924 |
| Mean prompt tokens / request | 48,119 | 33,537 |
| Generated tokens across completed requests | 58,740 | 149,697 |
| Mean generated tokens / request | 625 | 624 |
| Request-level errors | 3 | 0 |

The acceptance difference is 25.6551 percentage points, or 2.886 times by
ratio. These are valid per-run measurements from before/after Prometheus
counter snapshots. They are not yet a controlled comparison: the PP4/TP4 run
completed 94 requests while TP8 completed 240, and the mean prompt lengths
differ. The test operator should confirm whether both runs used exactly the
same input records and request options. Do not infer a deployment-only effect
from these aggregates.

Unconditional acceptance means accepted at position N divided by *all* draft
rounds. Both runs proposed exactly eight tokens per round, so the position
denominator is unambiguous.

| Draft position | PP4/TP4 accepted | PP4/TP4 rate | TP8 accepted | TP8 rate |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 13,218 | 47.0057% | 28,607 | 79.0729% |
| 2 | 7,662 | 27.2475% | 22,262 | 61.5346% |
| 3 | 4,337 | 15.4232% | 17,317 | 47.8661% |
| 4 | 2,416 | 8.5917% | 13,643 | 37.7108% |
| 5 | 1,382 | 4.9147% | 10,919 | 30.1813% |
| 6 | 805 | 2.8627% | 8,671 | 23.9676% |
| 7 | 476 | 1.6927% | 6,889 | 19.0420% |
| 8 | 298 | 1.0597% | 5,305 | 14.6636% |

For diagnosis, the conditional continuation rates at positions 1–8 were
approximately 47%, 58%, 57%, 56%, 57%, 58%, 59%, 63% on PP4/TP4 and
79%, 78%, 78%, 79%, 80%, 79%, 79%, 77% on TP8. The gap begins at the first
draft token. The zero-acceptance share was 52.99% versus 20.93%, and the
full-eight acceptance share was 1.06% versus 14.66%.

## Configuration differences to control

| Setting | PP4/TP4 disaggregated | TP8 colocated |
| --- | --- | --- |
| Prefill | TP1, PP4, layers 22/20/20/16 | Same TP8 engine |
| Decode | TP4, PP1 | TP8, PP1 |
| KV handoff | NIXL push | No handoff |
| Draft model | dSpark, eight tokens, draft TP1 | Same |
| Confidence scheduling | Disabled | Disabled |
| Decode async scheduling | Enabled | Enabled |
| Prefix caching | Enabled | Enabled |
| Expert parallel | Disabled | Disabled |
| Decode max sequences | 32 | 32 |
| Decode max batched tokens | 256 | 32,768 |
| KV block size / dtype | 64 / BF16 | 64 / BF16 |
| Attention backend | FLASHINFER_MLA_SPARSE | FLASHINFER_MLA_SPARSE |
| CUDA graph mode observed | FULL_DECODE_ONLY | FULL_AND_PIECEWISE |
| API path | P/D proxy | Direct server |

The different decode TP size, prompt mix, scheduling budget, graph mode, and
NIXL handoff all changed together. Keep these variables explicit in every
follow-up result.

The PP4/TP4 run logged three xgrammar structured-output rejections. It did
not log NIXL transfer failures in the measurement window. The TP8 run logged
no request-level errors. Three errors among 94 requests are material for the
request-success comparison, but do not alone explain the acceptance-rate
difference across more than 28,000 draft rounds.

## Highest-priority code hypothesis: draft KV transfer layout

This is a concrete, **unconfirmed** hypothesis. The target model uses MLA KV,
which is copied to each decode rank. The dSpark draft uses ordinary
head-sharded attention KV. Only the final PP prefill stage contains the mixed
target-MLA plus draft-full-attention KV regions.

In the current branch, NIXL prepares one source descriptor for each TP4
decode rank by dividing a draft cache block into four equal contiguous byte
ranges. See NixlBaseConnectorWorker._build_local_splits_from_plan in
../../vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py.
Both prefill and decode startup logs reported the LBNHC cache layout.
KVCacheLayout defines LBNHC as [layer, block, token, head, channel] and
LBHNC as [layer, block, head, token, channel] in
../../vllm/v1/kv_cache_layout.py. In LBNHC, one head's bytes are generally
interleaved across tokens. A contiguous quarter of the block therefore need
not equal one quarter of the heads.

The existing heterogeneous-TP layout guard requires contiguous heads, but
checks the model-wide use_mla flag. That flag is true for this mixed model,
so the guard does not reject its ordinary-attention draft region. Inspect
NixlBaseConnectorWorker._validate_remote_agent_handshake in the same file.
The descriptor length check validates byte counts, not head identity. A
successful NIXL transfer therefore does not prove the draft KV was mapped to
the right heads and token offsets.

If this suspicion is correct, the first draft step after remote prefill may
read incorrect draft history, reducing acceptance even while target inference
and NIXL transfer completion look healthy. A direct tensor comparison is
needed before treating this as the cause.

## Reproduction sequence for the next machine

1. Fix one input set. Save a manifest of request IDs, prompt token counts,
   generation limits, sampling parameters, tool/structured-output options,
   and a cryptographic hash of each request body. Keep raw prompts outside
   this repository. Replay the same records in the same order to every mode.
2. Use the same vLLM commit, target/draft checkpoints, tokenizer, numerical
   precision, eight speculative tokens, draft TP1, and confidence setting.
   Capture the full resolved engine config from startup logs. Make decode
   max batched tokens and graph mode equal where supported.
3. Run colocated TP4 on the same four GPUs used by the disaggregated decoder.
   This controls TP size. Compare it with colocated TP8 before investigating
   NIXL.
4. On the TP4 decoder, compare the same prompt sent directly to its local
   API (local prefill) with the PP4-prefill-to-TP4-decode route. Keep sampling
   deterministic. If direct TP4 acceptance is high but remote TP4 acceptance
   is low, focus on KV handoff and PP draft state rather than TP size.
5. For a small prompt with known blocks, instrument the final prefill stage
   immediately before NIXL push and every decode rank immediately after KV
   receipt. Compare draft KV by logical token, KV head, and channel, not by
   flat byte offset alone. Compare target MLA and draft regions separately.
   Record region names, tensor shapes, strides, cache layout, source and
   destination TP ranks, byte lengths, and per-head hashes. Do not log raw
   prompt content or arbitrary tensor values.
6. If draft KV differs, try a supported head-contiguous transfer layout or
   an explicit head-aware gather/permute buffer. Verify the copied tensor
   against local TP4 prefill before running another acceptance benchmark.
7. If KV matches, compare draft token IDs and target logits at the first
   decode step for a matched request. Then inspect PP draft state broadcast,
   proposal alignment, prefix-cache hits, graph mode, and asynchronous batch
   ordering.

Record a before/after snapshot of the following Decode counters for every
run: spec_decode_num_drafts_total, spec_decode_num_draft_tokens_total,
spec_decode_num_accepted_tokens_total,
spec_decode_num_accepted_tokens_per_pos_total (positions 0–7),
request_prompt_tokens_count/sum, request_generation_tokens_count/sum, and
request_success_total by finish reason. Report deltas; a service restart
resets these counters. The existing launcher and proxy in this directory
provide generic PP4/TP4 configuration without machine-specific addresses.

## Decision table for the first controlled rerun

| Observation | Most useful next check |
| --- | --- |
| TP4 colocated is already low | TP4 numerics, draft TP1 interaction, graph mode, workload |
| TP4 colocated is high, remote TP4 is low | Draft/target KV transfer or PP state handoff |
| Draft KV differs by head or token after transfer | LBNHC descriptor slicing and layout conversion |
| Draft KV matches, first-step proposals differ | PP draft state, sampling inputs, prefix cache, batch order |
| Matched requests converge across modes | Original gap was driven by unmatched traffic or settings |

No root cause has been proven yet. The first controlled replay and the
per-head KV comparison are the key evidence to collect.
