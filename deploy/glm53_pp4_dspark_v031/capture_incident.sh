#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
out=${1:?usage: capture_incident.sh OUT_DIR}
mkdir -p "$out"
journalctl -u glm53-mg43-pp4-p -u glm53-mg43-pp4-d \
  -u glm53-mg43-pp4-router -f -o short-iso-precise --since now \
  > "$out/journal.log" &
journal_pid=$!
trap 'kill "$journal_pid" 2>/dev/null || true' EXIT
while :; do
  {
    date --iso-8601=seconds
    for port in 8110 8111 8112; do
      code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$port/health" || true)
      echo "health_$port=$code"
    done
    for port in 8111 8112; do
      curl -fsS --max-time 3 "http://127.0.0.1:$port/metrics" 2>/dev/null \
        | grep -E '^vllm:(num_requests_(running|waiting|swapped)|num_skipped_waiting_reqs|kv_cache_usage_perc|num_preemptions|request_success_total|kv_connector_)' \
        | sed "s/^/metrics_$port /" || true
    done
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
      | sed 's/^/gpu /' || true
  } >> "$out/health_metrics.log"
  sleep 10
done
