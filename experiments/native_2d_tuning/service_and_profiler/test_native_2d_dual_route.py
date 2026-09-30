#!/usr/bin/env python3
"""Check the opt-in Iluvatar service route against a paged-KV reference."""

from __future__ import annotations

import json
import os

import torch

from native_2d_harness import batch, kwargs, reference
from vllm_fl.dispatch.backends.vendor.iluvatar.impl import attention


def main() -> None:
    os.environ["ILUVATAR_NATIVE_2D_DUAL"] = "1"
    device = torch.device("cuda:2")
    torch.cuda.set_device(device)
    qs = [2048] + [1] * 30
    prefixes = [8192] + [16384] * 30
    q, cache, cu, lens, table = batch(device, qs, prefixes, 20260929)
    out = torch.empty_like(q)
    call = kwargs(q, cache, cu, lens, table, out)
    call["max_seqlen_q"] = max(qs)

    captured = []
    original = attention.native_tuned_attention

    def capture(**arguments):
        captured.append(arguments["mixed_dual_launch_override"])
        return original(**arguments)

    attention.native_tuned_attention = capture
    attention._run_unified_attention(**call)
    torch.cuda.synchronize()
    error = (out.float() - reference(q, cache, qs, prefixes, table).float()).abs()
    if captured != [True] or error.max().item() > 0.05:
        raise AssertionError((captured, error.max().item()))

    # Unsupported semantics must stay on the existing tuned single launch.
    fallback = dict(call, sinks=torch.zeros((q.shape[1],), device=device))
    attention._run_unified_attention(**fallback)
    torch.cuda.synchronize()
    if captured != [True, False]:
        raise AssertionError(captured)
    print(json.dumps({"route_flags": captured, "max_abs": error.max().item()}))


if __name__ == "__main__":
    main()
