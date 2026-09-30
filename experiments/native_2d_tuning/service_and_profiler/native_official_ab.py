#!/usr/bin/env python3
"""Same-window official-load A/B for the opt-in native 2D candidate.

Only port 9032/GPU 1 is restarted.  The official 9031 service is checked but
never stopped or queried by the benchmark.  The candidate and baseline use
the same source tree; the marker alone selects native 2D tuned routing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request


ROOT = Path("/workspace/vllm-plugin-FL")
MODEL = "/workspace/MiniCPM5-2B"
MARKER = Path("/tmp/iluvatar_native_2d.enable")
CASES = [[4096, 1024, 64, 256], [16384, 1024, 64, 128]]


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def verify_9032(pid: int) -> None:
    command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    if "/vllm serve " not in command or "--port 9032" not in command:
        raise RuntimeError(f"PID {pid} is not the expected 9032 server: {command}")


def stop(pid: int) -> None:
    verify_9032(pid)
    os.kill(pid, signal.SIGTERM)
    for _ in range(180):
        stat = Path(f"/proc/{pid}/stat")
        if not stat.exists() or stat.read_text().split()[2] == "Z":
            return
        time.sleep(1)
    raise RuntimeError(f"9032 PID {pid} did not terminate cleanly")


def runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "1",
        "VLLM_PLUGINS": "fl",
        "PYTHONPATH": f"{ROOT}:/usr/local/lib/python3.12/site-packages",
        "LD_LIBRARY_PATH": ":".join(filter(None, (
            "/usr/local/corex/lib64", "/usr/local/openmpi/lib", "/usr/local/lib",
            env.get("LD_LIBRARY_PATH", ""),
        ))),
    })
    return env


def start(candidate: bool, log_path: Path) -> tuple[int, object]:
    if candidate:
        MARKER.touch()
    else:
        MARKER.unlink(missing_ok=True)
    command = [
        "/usr/local/bin/vllm", "serve", MODEL, "--port", "9032",
        "--served-model-name", "minicpm-diag", "--gpu-memory-utilization", "0.85",
        "--max-model-len", "131072", "--compilation-config",
        '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
    ]
    output = log_path.open("w")
    process = subprocess.Popen(command, cwd=ROOT, env=runtime_env(),
                               stdout=output, stderr=subprocess.STDOUT,
                               start_new_session=True)
    for _ in range(450):
        if process.poll() is not None:
            output.close()
            raise RuntimeError(f"9032 {('candidate' if candidate else 'baseline')} exited")
        if healthy(9032):
            return process.pid, output
        time.sleep(2)
    output.close()
    raise TimeoutError(f"9032 {('candidate' if candidate else 'baseline')} did not become healthy")


def benchmark(variant: str, work: Path) -> list[dict[str, str]]:
    work.mkdir(parents=True, exist_ok=True)
    command = [
        "/usr/local/bin/python3.12", "-u", str(ROOT / "benchmarks/benchmark_throughput_serve.py"),
        "--served-model-name", "minicpm-diag", "--model", MODEL, "--port", "9032",
        "--test-cases", json.dumps(CASES),
    ]
    with (work / "benchmark.log").open("w") as output:
        subprocess.run(command, cwd=work, env=runtime_env(), stdout=output,
                       stderr=subprocess.STDOUT, check=True, timeout=10800)
    summaries = list((work / "benchmark_results").glob("summary_*.csv"))
    raws = list((work / "benchmark_results").glob("raw_runs_*.csv"))
    if len(summaries) != 1 or len(raws) != 1:
        raise RuntimeError(f"{variant}: expected one raw and summary CSV")
    with summaries[0].open(newline="") as source:
        summary = list(csv.DictReader(source))
    with raws[0].open(newline="") as source:
        raw = list(csv.DictReader(source))
    if len(summary) != len(CASES) or len(raw) != 4 * len(CASES):
        raise RuntimeError(f"{variant}: incomplete CSV {len(summary)} / {len(raw)}")
    for case in CASES:
        rows = [r for r in raw if int(r["Prefill"]) == case[0] and int(r["Conc"]) == case[2]]
        if len(rows) != 4 or any(
            r["Run Status"] != "SUCCESS" or float(r["Successful Requests"]) != case[3]
            for r in rows[1:]
        ):
            raise RuntimeError(f"{variant}: failed steady-state case {case}")
    print(f"SUMMARY {variant}: {json.dumps(summary)}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-pid", type=int, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    verify_9032(args.expected_pid)
    if not healthy(9031) or not healthy(9032):
        raise RuntimeError("9031 and 9032 must be healthy before A/B")
    current_pid = args.expected_pid
    handle = None
    results: dict[str, list[dict[str, str]]] = {}
    try:
        for candidate, name in ((False, "baseline"), (True, "candidate")):
            stop(current_pid)
            if handle is not None:
                handle.close()
            current_pid, handle = start(candidate, args.log_dir / f"{name}_service.log")
            results[name] = benchmark(name, args.log_dir / name)
            (args.log_dir / "results.json").write_text(json.dumps(results, indent=2))
    finally:
        if current_pid is not None:
            stop(current_pid)
        if handle is not None:
            handle.close()
        MARKER.touch()
        restored_pid, restored_handle = start(True, args.log_dir / "restored_candidate_service.log")
        (args.log_dir / "restored_pid").write_text(f"{restored_pid}\n")
        restored_handle.close()
        print(f"RESTORED candidate pid={restored_pid}", flush=True)


if __name__ == "__main__":
    main()
