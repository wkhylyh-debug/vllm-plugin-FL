#!/usr/bin/env bash
set -uo pipefail

log_dir=${1:-/workspace/logs/native_2d_followup_20260929}
shape=${2:-mixed}
case "$shape" in
  mixed) stem=staged_scalar_smoke ;;
  mixed-order) stem=native_2d_mixed_order_smoke ;;
  pure) stem=staged_scalar_pure_smoke ;;
  early) stem=native_2d_early_smoke ;;
  *) printf 'Unsupported shape: %s\n' "$shape" >&2; exit 2 ;;
esac
extra=()
if [[ ${3:-} == audit ]]; then
  stem=native_2d_${shape}_error_audit
  extra=(--limit 1)
elif [[ ${3:-} == dual && ( $shape == mixed || $shape == mixed-order ) ]]; then
  stem=native_2d_${shape//-/_}_dual_smoke
  extra=(--mixed-dual-launch --limit 1)
fi
mkdir -p "$log_dir"
export PYTHONPATH=/workspace/vllm-plugin-FL:/usr/local/lib/python3.12/site-packages
export VLLM_PLUGINS=fl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

nice -n 15 perl -e 'alarm shift; exec @ARGV' 600 \
  python3 -u /workspace/vllm-plugin-FL/tools/native_2d_harness.py \
  --device cuda:2 --shape "$shape" --block-m 128 --tile 16 \
  --warps 8 --stages 2 --pipeline-stages 1 2 \
  --scalar-block-lookup 0 1 --warmup 2 --iterations 5 \
  "${extra[@]}" \
  --result "$log_dir/$stem.json" \
  >"$log_dir/$stem.log" 2>&1
rc=$?
printf '%s\n' "$rc" >"$log_dir/$stem.exit"
exit "$rc"
