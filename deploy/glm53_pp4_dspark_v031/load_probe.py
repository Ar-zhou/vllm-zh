#!/usr/bin/env python3
"""Bounded streaming load probe for the PP/Mooncake incident."""
import argparse
import concurrent.futures
import json
import threading
import time
import urllib.error
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8110/v1/chat/completions")
parser.add_argument("--concurrency", type=int, default=16)
parser.add_argument("--minutes", type=float, default=20)
parser.add_argument("--max-tokens", type=int, default=4096)
parser.add_argument("--min-tokens", type=int, default=0)
parser.add_argument("--max-errors", type=int, default=3)
parser.add_argument("--out", required=True)
args = parser.parse_args()
model = "glm-5.3-dspark"
lock = threading.Lock()
stop = threading.Event()
deadline = time.monotonic() + args.minutes * 60
started = time.monotonic()
counters = {"next_id": 0, "success": 0, "error": 0}


def one_request(request_id: int) -> dict:
    prompt = (
        (f"背景资料编号{request_id}，甲乙丙丁戊己庚辛。" * 160)
        + f"任务编号 {request_id}。请从1开始连续列出整数，为每个整数写一句独立的数学事实。"
        "持续写到输出上限，不要使用省略号，不要提前总结。"
    )
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "min_tokens": args.min_tokens,
        "ignore_eos": args.min_tokens > 0,
        "temperature": 0.3,
        "stream": True,
        "reasoning_effort": "low",
        "chat_template_kwargs": {"enable_thinking": True},
    }).encode()
    req = urllib.request.Request(args.url, data=body,
                                 headers={"Content-Type": "application/json"})
    started_at = time.time()
    chunks = 0
    status = None
    finish = None
    error = None
    try:
        with urllib.request.urlopen(req, timeout=900) as response:
            status = response.status
            for raw in response:
                if not raw.startswith(b"data: "):
                    continue
                payload = raw[6:].strip()
                if payload == b"[DONE]":
                    break
                event = json.loads(payload)
                if event.get("choices"):
                    chunks += 1
                    finish = event["choices"][0].get("finish_reason") or finish
                elif event.get("error"):
                    raise RuntimeError(str(event["error"]))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "id": request_id, "start": started_at, "end": time.time(),
        "status": status, "chunks": chunks, "finish": finish, "error": error,
    }


def worker(handle):
    while not stop.is_set() and time.monotonic() < deadline:
        with lock:
            request_id = counters["next_id"]
            counters["next_id"] += 1
        row = one_request(request_id)
        with lock:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if row["error"] or row["status"] != 200:
                counters["error"] += 1
                print(f"ERROR id={request_id} {row['error']}", flush=True)
                if counters["error"] >= args.max_errors:
                    stop.set()
            else:
                counters["success"] += 1
                if counters["success"] % 10 == 0:
                    print(f"success={counters['success']} elapsed={time.monotonic()-started:.0f}s", flush=True)


with open(args.out, "a", encoding="utf-8", buffering=1) as handle:
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(worker, handle) for _ in range(args.concurrency)]
        try:
            for future in concurrent.futures.as_completed(futures):
                future.result()
        except KeyboardInterrupt:
            stop.set()
print(counters, flush=True)
