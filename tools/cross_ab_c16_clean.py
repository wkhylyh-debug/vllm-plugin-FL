#!/usr/bin/env python3
"""Crossed c16 service A/B with prefix caching disabled.

This is an experiment harness only. It uses port 9032/GPU 1 and never
touches the official 9031 service.
"""

import datetime
import json
import os
import pathlib
import signal
import subprocess
import time
import urllib.request


LOGDIR = pathlib.Path(os.environ.get("LOGDIR", "/workspace/logs/cross_ab_c16_clean_20260929"))
ROOT = "/workspace/vllm-plugin-FL"
CANDIDATE_ROOT = "/workspace/split_kv_repair_20260928"
MODEL = "/workspace/MiniCPM5-2B"
CASE = "[[16384,256,16,16]]"
PAIRS = int(os.environ.get("AB_PAIRS", "6"))
SPLITS = int(os.environ.get("ILUVATAR_NUM_SPLITS", "4"))
BLOCK_M = int(os.environ.get("ILUVATAR_BLOCK_M", "128"))
BLOCK_N = int(os.environ.get("ILUVATAR_BLOCK_N", "64"))


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def write(message):
    print(f"[{now()}] {message}", flush=True)


def port_pids():
    output = subprocess.run(
        ["ps", "-eo", "pid=,args="], text=True, capture_output=True, check=False
    ).stdout
    result = []
    for line in output.splitlines():
        line = line.strip()
        if "--port 9032" in line and "vllm serve" in line:
            try:
                result.append(int(line.split(None, 1)[0]))
            except (ValueError, IndexError):
                pass
    return sorted(set(result))


def stop_9032():
    pids = port_pids()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 120
    while time.time() < deadline and port_pids():
        time.sleep(1)
    for pid in port_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if port_pids():
        raise RuntimeError(f"9032 still has processes: {port_pids()}")


def healthy():
    try:
        with urllib.request.urlopen("http://127.0.0.1:9032/health", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def start_variant(variant, service_log):
    env = os.environ.copy()
    env.update({"VLLM_PLUGINS": "fl", "CUDA_VISIBLE_DEVICES": "1"})
    if variant == "candidate":
        env.update(
            {
                "PYTHONPATH": f"{CANDIDATE_ROOT}:{ROOT}:/usr/local/lib/python3.12/site-packages",
                "ILUVATAR_USE_OPTIMIZED": "1",
                "ILUVATAR_SPLIT_KV": "1",
                "ILUVATAR_TRACE_ROUTES": "1",
                "ILUVATAR_NUM_SPLITS": str(SPLITS),
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
        "9032",
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
    write(f"started {variant} pid={process.pid}")
    deadline = time.time() + 900
    while time.time() < deadline:
        if healthy():
            write(f"{variant} healthy pid={process.pid}")
            return process, handle
        if process.poll() is not None:
            raise RuntimeError(f"{variant} exited rc={process.returncode}; see {service_log}")
        time.sleep(2)
    raise TimeoutError(f"{variant} did not become healthy; see {service_log}")


def run_benchmark(variant, pair, bench_log):
    command = [
        "/usr/local/bin/python3",
        "-u",
        f"{ROOT}/benchmarks/benchmark_throughput_serve.py",
        "--served-model-name",
        "minicpm-diag",
        "--model",
        MODEL,
        "--port",
        "9032",
        "--test-cases",
        CASE,
    ]
    env = os.environ.copy()
    env.update(
        {
            "VLLM_PLUGINS": "fl",
            "PYTHONPATH": f"{ROOT}:/usr/local/lib/python3.12/site-packages",
        }
    )
    with open(bench_log, "w") as handle:
        write(f"benchmark {variant} pair={pair} log={bench_log}")
        result = subprocess.run(
            command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    write(f"benchmark {variant} pair={pair} rc={result.returncode}")
    if result.returncode != 0:
        raise RuntimeError(f"benchmark failed rc={result.returncode}; see {bench_log}")


def main():
    LOGDIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    stop_9032()
    for pair in range(1, PAIRS + 1):
        order = [("baseline", None), ("candidate", None)] if pair % 2 else [("candidate", None), ("baseline", None)]
        for variant, _ in order:
            stop_9032()
            stem = f"pair{pair:02d}_{variant}"
            service_log = LOGDIR / f"{stem}_service.log"
            bench_log = LOGDIR / f"{stem}_bench.log"
            handle = None
            try:
                _, handle = start_variant(variant, service_log)
                run_benchmark(variant, pair, bench_log)
                manifest.append(
                    {
                        "pair": pair,
                        "variant": variant,
                        "service_log": str(service_log),
                        "bench_log": str(bench_log),
                        "rc": 0,
                    }
                )
            finally:
                if handle is not None:
                    handle.flush()
                    handle.close()
                stop_9032()
            (LOGDIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write("clean cross A/B complete")


if __name__ == "__main__":
    main()
