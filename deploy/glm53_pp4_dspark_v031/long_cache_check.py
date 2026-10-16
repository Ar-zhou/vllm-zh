#!/usr/bin/env python3
"""Exercise heterogeneous PP KV cache store and reload through the router.

Both runs send byte-identical requests. The first one populates LMCache from
every pipeline stage, the second one has to reload the same KV blocks. With
LMCache's MLA default `save_only_first_rank: true` the later PP stages store
nothing, the reload returns zero tokens, and the second run dies with a KV
load error, so a clean second run is the regression signal for this deploy.

`max_tokens` is deliberately generous: the prompt enables thinking, and a
limit that only covers the reasoning leaves the answer body empty. Override
it with the MAX_TOKENS environment variable.
"""
import json
import os
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8110/v1/chat/completions"
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))
body = json.dumps(
    {
        "model": "GLM-5.3-W4A4C8-MG43-BNT3-1P1D",
        "messages": [
            {
                "role": "user",
                "content": ("alpha beta gamma delta. " * 2300)
                + "\n请用一句话回答：1+1等于几？",
            }
        ],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": True},
    }
).encode()
failures = 0
for run in (1, 2):
    start = time.monotonic()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    content = []
    reasoning = []
    finish = None
    chunks = 0
    status = None
    error = ""
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            status = response.status
            for raw in response:
                if not raw.startswith(b"data: "):
                    continue
                payload = raw[6:].strip()
                if payload == b"[DONE]":
                    break
                event = json.loads(payload)
                if "choices" not in event:
                    raise RuntimeError(f"router event: {event}")
                choice = event["choices"][0]
                delta = choice.get("delta") or {}
                content.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning") or "")
                finish = choice.get("finish_reason") or finish
                chunks += 1
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = exc.read().decode(errors="replace")[:400]
    except Exception as exc:  # noqa: BLE001 - surface the failure, do not hide it
        error = f"{type(exc).__name__}: {exc}"
    if status != 200:
        failures += 1
    print(
        f"run={run} status={status} elapsed={time.monotonic()-start:.2f}s "
        f"chunks={chunks} reasoning_chars={len(''.join(reasoning))} "
        f"content={''.join(content)!r} finish={finish} error={error!r}",
        flush=True,
    )
raise SystemExit(1 if failures else 0)
