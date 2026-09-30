#!/usr/bin/env bash
set -euo pipefail

log_dir=${1:-/workspace/logs/native_2d_followup_20260929}
mkdir -p "$log_dir"

finish() {
  rc=$?
  printf '%s\n' "$rc" >"$log_dir/native_2d_followup_audits.exit"
}
trap finish EXIT

# Keep the c64 service comparison uncontended before using the other GPU.
for ((attempt = 0; attempt < 240; attempt++)); do
  if [[ -f $log_dir/baseline_c64.done ]]; then
    break
  fi
  sleep 30
done
if [[ ! -f $log_dir/baseline_c64.done ]]; then
  printf 'Baseline c64 did not finish within two hours\n' >&2
  exit 1
fi

for shape_mode in 'early audit' 'mixed dual' 'mixed-order dual'; do
  read -r shape mode <<<"$shape_mode"
  bash /workspace/vllm-plugin-FL/tools/run_native_2d_smoke.sh \
    "$log_dir" "$shape" "$mode"
done

python3 - "$log_dir" <<'PY'
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
for stem in (
    "native_2d_early_error_audit",
    "native_2d_mixed_dual_smoke",
    "native_2d_mixed_order_dual_smoke",
):
    data = json.loads((folder / f"{stem}.json").read_text())
    rows = data["results"]
    if not rows or any(row["status"] != "ok" for row in rows):
        raise SystemExit(f"{stem} has a failed correctness result")
    if "dual" in stem and any("dual_ms" not in row for row in rows):
        raise SystemExit(f"{stem} is missing dual-launch measurements")
    print(stem, [(row["shape"], row.get("dual_vs_tuned")) for row in rows])
PY
