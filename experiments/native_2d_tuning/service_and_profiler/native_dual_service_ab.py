#!/usr/bin/env python3
"""Isolated 9032 A/B for the opt-in Iluvatar mixed dual launch."""

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
DUAL_ROOT = Path("/workspace/native2d_gpu2_prototype_20260929")
MODEL = "/workspace/MiniCPM5-2B"
DUAL_MARKER = Path("/tmp/iluvatar_native_2d_dual.enable")
TUNED_MARKER = Path("/tmp/iluvatar_native_2d.enable")
CASES = [[4096, 128, 16, 16], [16384, 128, 16, 16],
         [4096, 128, 64, 64], [16384, 128, 64, 64]]


def health(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def verify_pid(pid: int) -> None:
    command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    if "/vllm serve " not in command or "--port 9032" not in command:
        raise RuntimeError(f"PID {pid} is not the expected 9032 server: {command}")


def stop(pid: int) -> None:
    verify_pid(pid)
    os.kill(pid, signal.SIGTERM)
    for _ in range(120):
        stat_path = Path(f"/proc/{pid}/stat")
        if not stat_path.exists() or stat_path.read_text().split()[2] == "Z":
            return
        time.sleep(1)
    raise RuntimeError(f"9032 server PID {pid} did not terminate cleanly")


def start(root: Path, log: Path, *, no_prefix_cache: bool) -> tuple[int, object]:
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "1", "VLLM_PLUGINS": "fl",
        "PYTHONPATH": f"{root}:/usr/local/lib/python3.12/site-packages",
        # Keep the Iluvatar runtime visible when the orchestrator is launched
        # from the host namespace or by a detached shell inside the container.
        "LD_LIBRARY_PATH": ":".join(filter(None, (
            "/usr/local/corex/lib64",
            "/usr/local/openmpi/lib",
            "/usr/local/lib",
            os.environ.get("LD_LIBRARY_PATH", ""),
        ))),
    })
    command = [
        "/usr/local/bin/vllm", "serve", MODEL, "--port", "9032",
        "--served-model-name", "minicpm-diag", "--gpu-memory-utilization", "0.85",
        "--max-model-len", "131072", "--compilation-config",
        '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
    ]
    if no_prefix_cache:
        command.append("--no-enable-prefix-caching")
    output = log.open("w")
    process = subprocess.Popen(command, cwd=root, env=environment,
                               stdout=output, stderr=subprocess.STDOUT,
                               start_new_session=True)
    print(f"START pid={process.pid} root={root} no_prefix_cache={no_prefix_cache}", flush=True)
    for _ in range(450):
        if process.poll() is not None:
            output.close()
            raise RuntimeError(f"9032 exited during startup: {log}")
        if health(9032):
            return process.pid, output
        time.sleep(2)
    output.close()
    raise RuntimeError(f"9032 did not become healthy: {log}")


def read_benchmark(name: str, work: Path) -> list[dict[str, str]]:
    log = work / "benchmark.log"
    summary_paths = list((work / "benchmark_results").glob("summary_*.csv"))
    raw_paths = list((work / "benchmark_results").glob("raw_runs_*.csv"))
    if len(summary_paths) != 1 or len(raw_paths) != 1:
        raise RuntimeError(f"Expected one raw and summary CSV for {name}: {log}")
    with summary_paths[0].open(newline="") as source:
        summary = list(csv.DictReader(source))
    with raw_paths[0].open(newline="") as source:
        raw = list(csv.DictReader(source))
    if len(summary) != len(CASES) or len(raw) != 4 * len(CASES):
        raise RuntimeError(f"Missing case/run for {name}: {len(summary)} / {len(raw)}")
    for case in CASES:
        rows = [r for r in raw if int(r["Prefill"]) == case[0]
                and int(r["Conc"]) == case[2]]
        if (len(rows) != 4 or any(
            r["Run Status"] != "SUCCESS" or float(r["Successful Requests"]) != case[3]
            for r in rows[1:]
        )):
            raise RuntimeError(f"Failed steady-state requests for {name}: {case}")
    print(f"SUMMARY {name}: " + json.dumps(summary), flush=True)
    return summary


def benchmark(name: str, directory: Path) -> list[dict[str, str]]:
    work = directory / name
    work.mkdir()
    env = os.environ.copy()
    env.update({"VLLM_PLUGINS": "fl",
                "PYTHONPATH": f"{ROOT}:/usr/local/lib/python3.12/site-packages",
                "LD_LIBRARY_PATH": ":".join(filter(None, (
                    "/usr/local/corex/lib64",
                    "/usr/local/openmpi/lib",
                    "/usr/local/lib",
                    os.environ.get("LD_LIBRARY_PATH", ""),
                )))})
    command = [
        "/usr/local/bin/python3.12", "-u",
        str(ROOT / "benchmarks/benchmark_throughput_serve.py"),
        "--served-model-name", "minicpm-diag", "--model", MODEL,
        "--port", "9032", "--test-cases", json.dumps(CASES),
    ]
    print(f"BENCHMARK {name}: {command}", flush=True)
    with (work / "benchmark.log").open("w") as output:
        subprocess.run(command, cwd=work, env=env, stdout=output,
                       stderr=subprocess.STDOUT, check=True, timeout=7200)
    return read_benchmark(name, work)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-pid", type=int, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--reuse-tuned-dir", type=Path)
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    verify_pid(args.expected_pid)
    if not health(9031) or not health(9032):
        raise RuntimeError("9031 and 9032 must both be healthy before the A/B")
    if not ROOT.is_dir() or not DUAL_ROOT.is_dir():
        raise RuntimeError("Pinned candidate or isolated prototype is missing")
    if not TUNED_MARKER.exists():
        raise RuntimeError("Tuned marker is missing; baseline route is ambiguous")
    current_pid = args.expected_pid
    log_handle = None
    results = {}
    if args.reuse_tuned_dir is not None:
        results["tuned"] = read_benchmark("tuned", args.reuse_tuned_dir)
    try:
        variants = (("dual", DUAL_ROOT),) if args.reuse_tuned_dir else (
            ("tuned", ROOT), ("dual", DUAL_ROOT)
        )
        for name, root in variants:
            stop(current_pid)
            current_pid = None
            if log_handle:
                log_handle.close()
            if name == "dual":
                DUAL_MARKER.touch()
            else:
                DUAL_MARKER.unlink(missing_ok=True)
            current_pid, log_handle = start(root, args.log_dir / f"{name}_service.log",
                                            no_prefix_cache=True)
            results[name] = benchmark(name, args.log_dir)
            (args.log_dir / "results.json").write_text(json.dumps(results, indent=2))
    finally:
        if current_pid is not None:
            stop(current_pid)
        if log_handle:
            log_handle.close()
        DUAL_MARKER.unlink(missing_ok=True)
        restored_pid, restored_log = start(ROOT, args.log_dir / "restored_service.log",
                                            no_prefix_cache=False)
        (args.log_dir / "restored_pid").write_text(f"{restored_pid}\n")
        restored_log.close()
        print(f"RESTORED pinned candidate pid={restored_pid}", flush=True)
    if not health(9031):
        raise RuntimeError("9031 health check failed after A/B")


if __name__ == "__main__":
    main()
