#!/usr/bin/env bash
set -Eeuo pipefail

role=${1:?usage: stop_engine.sh p|d}
case "$role" in p|d) ;; *) echo "role must be p or d" >&2; exit 2 ;; esac
CONTAINERD_NAMESPACE=${CONTAINERD_NAMESPACE:-default}
CONTAINER_PREFIX=${CONTAINER_PREFIX:-pp4-swa-pd}
name="$CONTAINER_PREFIX-$role"

ctr -n "$CONTAINERD_NAMESPACE" tasks kill --all --signal SIGTERM "$name" 2>/dev/null || true
for _ in {1..30}; do
  if ! ctr -n "$CONTAINERD_NAMESPACE" tasks list -q 2>/dev/null | grep -Fxq "$name"; then
    break
  fi
  sleep 1
done
ctr -n "$CONTAINERD_NAMESPACE" tasks kill --all --signal SIGKILL "$name" 2>/dev/null || true
ctr -n "$CONTAINERD_NAMESPACE" tasks delete "$name" 2>/dev/null || true
ctr -n "$CONTAINERD_NAMESPACE" containers delete "$name" 2>/dev/null || true
