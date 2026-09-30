#!/usr/bin/env bash
set -euo pipefail

log_dir=${1:-/workspace/logs/native_2d_followup_20260929}
baseline_pid=${2:?explicit 9032 baseline PID required}
pin="$log_dir/candidate_pinned/vllm_fl/dispatch/backends/vendor/iluvatar/impl"
impl=/workspace/vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl
mkdir -p "$log_dir"

finish() {
  rc=$?
  printf '%s\n' "$rc" >"$log_dir/native_2d_level3.exit"
}
trap finish EXIT

for ((attempt = 0; attempt < 240; attempt++)); do
  if [[ -f $log_dir/baseline_c64.done &&
        -f $log_dir/native_2d_followup_audits.exit ]]; then
    break
  fi
  sleep 30
done
if [[ ! -f $log_dir/baseline_c64.done ||
      ! -f $log_dir/native_2d_followup_audits.exit ]]; then
  printf 'c64 comparison or GPU 2 audit did not finish in time\n' >&2
  exit 1
fi

# The uncommitted dual-launch experiment is not part of this accuracy run.
# Its speed or compilation failure cannot affect the pinned candidate. The
# pinned candidate must, however, pass the early-prefill numerical audit.
python3 - "$log_dir/native_2d_early_error_audit.json" <<'PY'
import json
import sys

rows = json.load(open(sys.argv[1]))["results"]
if len(rows) != 3 or any(row["status"] != "ok" for row in rows):
    raise SystemExit("candidate early-prefill correctness check failed")
PY

cmdline=$(tr '\0' ' ' <"/proc/$baseline_pid/cmdline")
if [[ $cmdline != *'/vllm serve '* || $cmdline != *'--port 9032'* ]]; then
  printf 'PID %s is not the expected 9032 server: %s\n' "$baseline_pid" "$cmdline" >&2
  exit 1
fi
kill -TERM "$baseline_pid"
for ((attempt = 0; attempt < 120; attempt++)); do
  if [[ ! -e /proc/$baseline_pid/cmdline ]]; then
    break
  fi
  state=$(ps -p "$baseline_pid" -o stat= 2>/dev/null || true)
  if [[ $state == Z* || -z $state ]]; then
    break
  fi
  sleep 5
done
if [[ -n $(ps -p "$baseline_pid" -o stat= 2>/dev/null || true) &&
      $(ps -p "$baseline_pid" -o stat= 2>/dev/null) != Z* ]]; then
  printf '9032 baseline did not stop cleanly\n' >&2
  exit 1
fi

install -m 0644 "$pin/attention.py" "$impl/attention.py"
install -m 0644 "$pin/ops/triton_unified_attention_native.py" \
  "$impl/ops/triton_unified_attention_native.py"
touch /tmp/iluvatar_native_2d.enable

env CUDA_VISIBLE_DEVICES=1 VLLM_PLUGINS=fl \
  PYTHONPATH=/workspace/vllm-plugin-FL:/usr/local/lib/python3.12/site-packages \
  /usr/local/bin/python3.12 /usr/local/bin/vllm serve \
  /workspace/MiniCPM5-2B --port 9032 --served-model-name minicpm-diag \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  >"$log_dir/candidate_level3_service.log" 2>&1 &
candidate_pid=$!
printf '%s\n' "$candidate_pid" >"$log_dir/candidate_level3_service.pid"

ready=0
for ((attempt = 0; attempt < 240; attempt++)); do
  if curl -fsS http://127.0.0.1:9032/health >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$candidate_pid" 2>/dev/null; then
    printf 'Candidate 9032 service exited before readiness\n' >&2
    exit 1
  fi
  sleep 5
done
if [[ $ready != 1 ]]; then
  printf 'Candidate 9032 service timed out waiting for health\n' >&2
  exit 1
fi

bash /workspace/vllm-plugin-FL/tools/stage1/level3_9032_command.sh \
  >"$log_dir/candidate_level3.log" 2>&1
touch "$log_dir/candidate_level3.done"
