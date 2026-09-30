#!/usr/bin/env python3
"""Profile the two mixed attention launches on an isolated GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from native_2d_harness import batch, invoke, kwargs, tuned_attention


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=True)

    shapes = {
        "mixed": ([2048] + [1] * 30, [8192] + [16384] * 30),
        "decode_first": ([1] * 30 + [2048], [16384] * 30 + [8192]),
        "interleaved": (
            [1] * 15 + [2048, 3] + [1] * 15,
            [16384] * 15 + [8192, 4096] + [16384] * 15,
        ),
    }
    records = []
    for name, (qs, prefixes) in shapes.items():
        q, cache, cu, lens, table = batch(device, qs, prefixes, args.seed)
        kw = kwargs(q, cache, cu, lens, table, torch.empty_like(q))
        common = dict(block_m=128, tile=16, warps=8, stages=2,
                      pipeline_stages=1, scalar_block_lookup=False)
        for variant, dual in (("tuned", False), ("dual", True)):
            call = dict(common, mixed_dual_launch=dual)
            for _ in range(5):
                invoke(tuned_attention, kw, max(qs), **call)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                with record_function(f"attention_{name}_{variant}"):
                    for _ in range(3):
                        invoke(tuned_attention, kw, max(qs), **call)
                    torch.cuda.synchronize()
            events = []
            for event in prof.events():
                if ("kernel_unified_attention" in event.name
                        or event.name.startswith("attention_")):
                    events.append({
                        "name": event.name,
                        "device_type": str(event.device_type),
                        "cpu_us": event.cpu_time_total,
                        "device_us": getattr(event, "device_time_total", None),
                    })
            if args.trace_dir:
                prof.export_chrome_trace(str(args.trace_dir / f"{name}_{variant}.json"))
            records.append({"shape": name, "variant": variant, "events": events})
            print(json.dumps(records[-1]), flush=True)
    args.output.write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
