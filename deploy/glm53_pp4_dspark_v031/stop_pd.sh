#!/usr/bin/env bash
set -Eeuo pipefail
name="glm53-mg43-pp4-${1:?p or d}"
ctr -n dspark-v41 tasks kill --all --signal SIGTERM "$name" 2>/dev/null || true
for _ in {1..30}; do
  if ! ctr -n dspark-v41 tasks list -q 2>/dev/null | grep -Fxq "$name"; then
    break
  fi
  sleep 1
done
ctr -n dspark-v41 tasks kill --all --signal SIGKILL "$name" 2>/dev/null || true
ctr -n dspark-v41 tasks delete "$name" 2>/dev/null || true
ctr -n dspark-v41 containers delete "$name" 2>/dev/null || true
