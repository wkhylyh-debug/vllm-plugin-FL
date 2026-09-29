#!/bin/sh
set -eu

log_path=${1:?log path is required}
cache_dir=${2:?cache directory is required}

mkdir -p "$(dirname "$log_path")" "$cache_dir"
touch /tmp/iluvatar_native_2d.enable

compilation_config=$(printf \
  '{"cudagraph_mode":"FULL_DECODE_ONLY","cache_dir":"%s"}' \
  "$cache_dir")

exec env \
  CUDA_VISIBLE_DEVICES=1 \
  PYTHONPATH=/workspace/vllm-plugin-FL:/usr/local/lib/python3.12/site-packages \
  /usr/local/bin/python3.12 /usr/local/bin/vllm serve \
  /workspace/MiniCPM5-2B \
  --port 9032 \
  --served-model-name minicpm-diag \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072 \
  --no-enable-prefix-caching \
  --compilation-config "$compilation_config" \
  >"$log_path" 2>&1
