#!/usr/bin/env python3
"""Bounded replay of CodeArts request bodies; record timing and usage only."""

import argparse
import concurrent.futures
import json
import threading
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8110/v1/chat/completions")
    parser.add_argument("--concurrency", type=int, default=34)
    parser.add_argument("--minutes", type=float, default=20)
    parser.add_argument("--max-records", type=int, default=200)
    parser.add_argument("--max-output-tokens", type=int, default=0,
                        help="Cap output only for a short precheck (0 keeps source value)")
    parser.add_argument("--min-observed-output-tokens", type=int, default=0,
                        help="Select rows whose earlier run generated at least this many tokens")
    parser.add_argument("--max-errors", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if min(args.concurrency, args.minutes, args.max_records, args.max_errors, args.timeout) <= 0:
        parser.error("numeric limits must be positive")
    if args.max_output_tokens < 0 or args.min_observed_output_tokens < 0:
        parser.error("output token limits must be nonnegative")

    jobs = []
    with open(args.source, encoding="utf-8") as source:
        for line_number, line in enumerate(source):
            source_row = json.loads(line)
            row = source_row["requestdata"]
            observed = ((source_row.get("response_data") or {}).get("usage") or {}).get(
                "completion_tokens"
            ) or 0
            if observed < args.min_observed_output_tokens:
                continue
            if (row.get("max_tokens") or 0) < 16_000:
                continue
            request_bytes = len(json.dumps(row, ensure_ascii=False).encode())
            if not 100_000 <= request_bytes <= 500_000:
                continue
            payload = {
                key: row[key]
                for key in ("messages", "max_tokens", "temperature", "top_p", "n",
                            "ignore_eos", "tools", "tool_choice", "parallel_tool_calls")
                if key in row
            }
            if not payload.get("tools"):
                for key in ("tools", "tool_choice", "parallel_tool_calls"):
                    payload.pop(key, None)
            payload.update(
                model="glm-5.3-dspark", stream=True,
                stream_options={"include_usage": True},
                chat_template_kwargs={"reasoning_effort": "max", "enable_thinking": True},
            )
            if args.max_output_tokens:
                payload["max_tokens"] = min(payload["max_tokens"], args.max_output_tokens)
            jobs.append((line_number, json.dumps(payload, ensure_ascii=False).encode()))
            if len(jobs) >= args.max_records:
                break
    print(f"selected={len(jobs)}", flush=True)
    if not jobs:
        raise SystemExit("No source rows match filter")

    lock = threading.Lock()
    stop = threading.Event()
    next_job = 0
    success = 0
    errors = 0
    deadline = time.monotonic() + args.minutes * 60

    def send(line_number, body):
        started = time.time()
        result = {"line_number": line_number, "start": started, "status": None,
                  "prompt_tokens": None, "completion_tokens": None,
                  "finish": None, "error": None}
        req = urllib.request.Request(args.url, data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as response:
                result["status"] = response.status
                done = False
                for raw in response:
                    if not raw.startswith(b"data: "):
                        continue
                    data = raw[6:].strip()
                    if data == b"[DONE]":
                        done = True
                        break
                    event = json.loads(data)
                    if event.get("error"):
                        raise RuntimeError(str(event["error"]))
                    usage = event.get("usage") or {}
                    for key in ("prompt_tokens", "completion_tokens"):
                        if usage.get(key) is not None:
                            result[key] = usage[key]
                    for choice in event.get("choices") or []:
                        result["finish"] = choice.get("finish_reason") or result["finish"]
                if not done:
                    raise RuntimeError("stream closed without [DONE]")
        except urllib.error.HTTPError as exc:
            result["status"] = exc.code
            result["error"] = f"HTTP {exc.code}: {exc.read(512).decode(errors='replace')}"
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["end"] = time.time()
        return result

    with open(args.out, "a", encoding="utf-8", buffering=1) as output:
        def worker():
            nonlocal next_job, success, errors
            while not stop.is_set() and time.monotonic() < deadline:
                with lock:
                    if next_job >= len(jobs):
                        return
                    line_number, body = jobs[next_job]
                    next_job += 1
                result = send(line_number, body)
                with lock:
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                    if result["error"] or result["status"] != 200:
                        errors += 1
                        print(f"ERROR line={line_number} {result['error']}", flush=True)
                        if errors >= args.max_errors:
                            stop.set()
                    else:
                        success += 1
                        if success % 10 == 0:
                            print(f"success={success} errors={errors}", flush=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for future in concurrent.futures.as_completed(
                [pool.submit(worker) for _ in range(args.concurrency)]
            ):
                future.result()
    print(f"selected={len(jobs)} started={next_job} success={success} errors={errors}", flush=True)


if __name__ == "__main__":
    main()
