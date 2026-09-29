#!/usr/bin/env python3
"""Run the minimum c16/c64 native-vs-Split-KV service matrix.

This harness only controls diagnostic port 9032 inside the mllv container.
The official 9031 service is never inspected beyond a health check and is
never stopped or sent requests.
"""

import datetime
import json
import os
import pathlib
import signal
import subprocess
import time
import urllib.request


ROOT = "/workspace/vllm-plugin-FL"
CANDIDATE_ROOT = "/workspace/split_kv_repair_20260928"
MODEL = "/workspace/MiniCPM5-2B"
PORT = 9032
LOGDIR = pathlib.Path(
    os.environ.get("LOGDIR", "/workspace/logs/cross_ab_matrix_20260928")
)
BLOCK_M = int(os.environ.get("ILUVATAR_BLOCK_M", "128"))
BLOCK_N = int(os.environ.get("ILUVATAR_BLOCK_N", "64"))

# The c16 native/candidate and c64 native/4/8 candidate cells are the five
# missing cells in the current evidence matrix. Each benchmark command runs
# four rounds and skips its first round as warmup.
CASES = [
    ("c16_native", "native", 16, 0),
    ("c16_4split", "candidate", 16, 4),
    ("c64_native", "native", 64, 0),
    ("c64_4split", "candidate", 64, 4),
    ("c64_8split", "candidate", 64, 8),
]


def stamp():
    return datetime.datetime.now().isoformat(timespec="seconds")


def log(message):
    print(f"[{stamp()}] {message}", flush=True)


def service_pids():
    output = subprocess.run(
        ["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=False
    ).stdout
    pids = []
    for line in output.splitlines():
        line = line.strip()
        if "vllm serve" in line and f"--port {PORT}" in line:
            try:
                pids.append(int(line.split(None, 1)[0]))
            except (ValueError, IndexError):
                pass
    return sorted(set(pids))


def stop_service():
    for pid in service_pids():
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 120
    while time.time() < deadline and service_pids():
        time.sleep(1)
    for pid in service_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if service_pids():
        raise RuntimeError(f"9032 still running: {service_pids()}")


def healthy():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def start_service(kind, splits, service_log):
    env = os.environ.copy()
    env.update({"VLLM_PLUGINS": "fl", "CUDA_VISIBLE_DEVICES": "1"})
    if kind == "candidate":
        env.update(
            {
                "PYTHONPATH": f"{CANDIDATE_ROOT}:{ROOT}:/usr/local/lib/python3.12/site-packages",
                "ILUVATAR_USE_OPTIMIZED": "1",
                "ILUVATAR_SPLIT_KV": "1",
                "ILUVATAR_TRACE_ROUTES": "1",
                "ILUVATAR_NUM_SPLITS": str(splits),
                "ILUVATAR_BLOCK_M": str(BLOCK_M),
                "ILUVATAR_BLOCK_N": str(BLOCK_N),
                "ILUVATAR_NUM_WARPS": "4",
                "ILUVATAR_NUM_STAGES": "2",
            }
        )
    else:
        env.update(
            {
                "PYTHONPATH": f"{ROOT}:/usr/local/lib/python3.12/site-packages",
                "ILUVATAR_USE_OPTIMIZED": "0",
                "ILUVATAR_SPLIT_KV": "0",
                "ILUVATAR_TRACE_ROUTES": "0",
            }
        )
    command = [
        "/usr/local/bin/vllm",
        "serve",
        MODEL,
        "--port",
        str(PORT),
        "--compilation-config",
        '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
        "--served-model-name",
        "minicpm-diag",
        "--gpu-memory-utilization",
        "0.85",
        "--max-model-len",
        "131072",
        "--no-enable-prefix-caching",
    ]
    handle = open(service_log, "w")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log(f"started {kind} splits={splits} pid={process.pid}")
    deadline = time.time() + 900
    while time.time() < deadline:
        if healthy():
            log(f"healthy {kind} pid={process.pid}")
            return handle
        if process.poll() is not None:
            raise RuntimeError(f"service exited rc={process.returncode}; see {service_log}")
        time.sleep(2)
    raise TimeoutError(f"service did not become healthy; see {service_log}")


def run_benchmark(case_name, concurrency, bench_log):
    command = [
        "/usr/local/bin/python3",
        "-u",
        f"{ROOT}/benchmarks/benchmark_throughput_serve.py",
        "--served-model-name",
        "minicpm-diag",
        "--model",
        MODEL,
        "--port",
        str(PORT),
        "--test-cases",
        json.dumps([[16384, 1024, concurrency, 128]]),
    ]
    env = os.environ.copy()
    env.update({"VLLM_PLUGINS": "fl", "PYTHONPATH": f"{ROOT}:/usr/local/lib/python3.12/site-packages"})
    with open(bench_log, "w") as handle:
        log(f"benchmark {case_name} log={bench_log}")
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    log(f"benchmark {case_name} rc={result.returncode}")
    if result.returncode != 0:
        raise RuntimeError(f"benchmark failed rc={result.returncode}; see {bench_log}")


def main():
    LOGDIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    stop_service()
    try:
        for case_name, kind, concurrency, splits in CASES:
            stop_service()
            service_log = LOGDIR / f"{case_name}_service.log"
            bench_log = LOGDIR / f"{case_name}_bench.log"
            handle = None
            try:
                handle = start_service(kind, splits, service_log)
                run_benchmark(case_name, concurrency, bench_log)
                manifest.append(
                    {
                        "case": case_name,
                        "kind": kind,
                        "concurrency": concurrency,
                        "splits": splits,
                        "service_log": str(service_log),
                        "bench_log": str(bench_log),
                        "rc": 0,
                    }
                )
            finally:
                if handle is not None:
                    handle.flush()
                    handle.close()
                stop_service()
            (LOGDIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        stop_service()
    log("matrix complete")


if __name__ == "__main__":
    main()
