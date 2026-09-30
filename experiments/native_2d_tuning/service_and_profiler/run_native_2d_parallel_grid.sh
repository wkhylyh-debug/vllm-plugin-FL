#!/usr/bin/env bash
set -euo pipefail

# Run disjoint native-2D launch-parameter scans in parallel. GPU 0/1 are
# reserved for the official and diagnostic services; the default workers use
# GPU 2-7 and expose each physical device as cuda:0 inside its process.

ROOT=${ROOT:-/workspace/vllm-plugin-FL}
OUT=${1:-/workspace/logs/native_2d_grid_$(date +%Y%m%d_%H%M%S)}
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ $# -gt 0 ]]; then
  GPUS=("$@")
else
  GPUS=(2 3 4 5 6 7)
fi

if [[ ${#GPUS[@]} -eq 0 ]]; then
  printf 'no worker GPUs supplied\n' >&2
  exit 2
fi

mkdir -p "$OUT"
printf 'output=%s\n' "$OUT"
printf 'workers=%s\n' "${GPUS[*]}"

run_worker() {
  local gpu=$1
  local result="$OUT/gpu${gpu}.json"
  local log="$OUT/gpu${gpu}.log"
  local args=()

  case "$gpu" in
    2)
      # Correctness-first scalar lookup ablation, including page-boundary
      # tails.  TILE_SIZE=8/16 are the only legal scalar configurations.
      args=(--shape early --block-m 128 --tile 8 16 --warps 8
            --stages 2 --pipeline-stages 1 --scalar-block-lookup 0 1
            --warmup 3 --iterations 8)
      ;;
    3)
      # TILE_SIZE scan with vector lookup.  This covers the long-context
      # candidates that cannot use scalar page-table lookup with BLOCK=16.
      args=(--shape both --block-m 128 --tile 16 32 64 --warps 8
            --stages 2 --pipeline-stages 1 --scalar-block-lookup 0
            --warmup 3 --iterations 8)
      ;;
    4)
      args=(--shape both --block-m 32 64 --tile 16 --warps 8
            --stages 2 --pipeline-stages 1 --scalar-block-lookup 0
            --warmup 3 --iterations 8)
      ;;
    5)
      args=(--shape both --block-m 128 256 --tile 16 --warps 8
            --stages 2 --pipeline-stages 1 --scalar-block-lookup 0
            --warmup 3 --iterations 8)
      ;;
    6)
      args=(--shape both --block-m 128 --tile 16 --warps 4 8
            --stages 1 2 --pipeline-stages 1 --scalar-block-lookup 0
            --warmup 3 --iterations 8)
      ;;
    7)
      args=(--shape both --block-m 128 --tile 16 --warps 16
            --stages 1 2 3 --pipeline-stages 1 --scalar-block-lookup 0
            --warmup 3 --iterations 8)
      ;;
    *)
      printf 'unsupported worker GPU %s\n' "$gpu" >&2
      return 2
      ;;
  esac

  printf 'GPU %s start: %s\n' "$gpu" "${args[*]}" | tee "$log"
  CUDA_VISIBLE_DEVICES="$gpu" \
    python3 "$ROOT/tools/native_2d_harness.py" \
      --device cuda:0 --result "$result" "${args[@]}" \
      >>"$log" 2>&1
  printf 'GPU %s done\n' "$gpu" >>"$log"
}

pids=()
for gpu in "${GPUS[@]}"; do
  run_worker "$gpu" &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done

printf 'completed output=%s rc=%s\n' "$OUT" "$rc"
exit "$rc"
