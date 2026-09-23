#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data_root="${GLM52_V030_DATA_ROOT:-/mnt/mtp/sfs/z00936200/glm52-pp8-v030}"
wheel_dir="$data_root/wheels"
wheel="$wheel_dir/vllm-0.30.0-cp38-abi3-manylinux_2_28_x86_64.whl"
pip_bin="${GLM52_V030_PIP_BIN:-/mnt/sfs_turbo/n30008093/.tools/python/cpython-3.12.13-linux-x86_64-gnu/bin/pip3.12}"

mkdir -p "$wheel_dir"
if [[ ! -f "$wheel" ]]; then
  "$pip_bin" download --no-deps --only-binary=:all: --dest "$wheel_dir" 'vllm==0.30.0'
fi
unzip -tq "$wheel" >/dev/null

# This checkout is source-only. Copy the matching release's ignored native
# extensions and vendored runtime into it, leaving tracked files untouched.
unzip -oq "$wheel" 'vllm/*.so' 'vllm/third_party/*' 'vllm/_version.py' -d "$repo_root"

"${GLM52_V030_PYTHON:-/mnt/sfs_turbo/n30008093/vllm/.venv/bin/python}" -c \
  'import sys; sys.path.insert(0, sys.argv[1]); import vllm; assert vllm.__version__ == "0.30.0", vllm.__version__; print(vllm.__version__, vllm.__file__)' \
  "$repo_root"
