#!/usr/bin/env python3
import json
import time
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
body = (BASE / "smoke_request.json").read_bytes()

for run in (1, 2):
    req = urllib.request.Request(
        "http://127.0.0.1:8110/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    content = []
    reasoning = []
    finish_reason = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=180) as response:
        status = response.status
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            payload = raw[6:].strip()
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            choice = chunk["choices"][0]
            delta = choice.get("delta") or {}
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning") or "")
            finish_reason = choice.get("finish_reason") or finish_reason
            chunks += 1
    print(
        f"run={run} status={status} elapsed={time.monotonic()-started:.2f}s "
        f"chunks={chunks} reasoning_chars={len(''.join(reasoning))} "
        f"content={''.join(content)!r} finish={finish_reason}",
        flush=True,
    )
    for port in (8111, 8112, 8110):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as check:
            print(f"health_{port}={check.status}", flush=True)
