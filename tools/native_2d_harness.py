#!/usr/bin/env python3
"""Screen native 2D attention launch parameters on a free Iluvatar GPU.

This harness keeps vLLM's request/KV ABI and compares the version-pinned
vendor copy with the installed vLLM kernel. It does not use the Split-KV or
Python request-partitioning path.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode
from vllm_fl.dispatch.backends.vendor.iluvatar.impl.ops.triton_unified_attention_native import (
    unified_attention as tuned_attention,
)


NQ = 16
NKV = 2
HD = 128
BLOCK = 16
DTYPE = torch.bfloat16


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--shape", choices=("pure", "mixed", "mixed-order", "early", "both"), default="both")
    p.add_argument("--block-m", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--tile", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--warps", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--pipeline-stages", type=int, nargs="+", default=[1])
    p.add_argument("--scalar-block-lookup", type=int, nargs="+", choices=(0, 1), default=[0])
    p.add_argument("--mixed-dual-launch", action="store_true")
    p.add_argument("--dual-first", action="store_true")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260929)
    return p.parse_args()


def batch(device: torch.device, qs: list[int], prefixes: list[int], seed: int):
    g = torch.Generator(device=device).manual_seed(seed)
    seqs = [q + p for q, p in zip(qs, prefixes)]
    blocks = [math.ceil(s / BLOCK) for s in seqs]
    cache = torch.randn(
        (sum(blocks), 2, BLOCK, NKV, HD), device=device, dtype=DTYPE, generator=g
    )
    table = torch.zeros((len(seqs), max(blocks)), device=device, dtype=torch.int32)
    cursor = 0
    for i, n in enumerate(blocks):
        table[i, :n] = torch.arange(cursor, cursor + n, device=device)
        cursor += n
    q = torch.randn((sum(qs), NQ, HD), device=device, dtype=DTYPE, generator=g)
    cu = torch.tensor([0, *torch.tensor(qs).cumsum(0).tolist()], device=device, dtype=torch.int32)
    lens = torch.tensor(seqs, device=device, dtype=torch.int32)
    return q, cache, cu, lens, table


def reference(q, cache, qs, prefixes, table):
    out = []
    off = 0
    scale = 1.0 / math.sqrt(HD)
    for i, (ql, pref) in enumerate(zip(qs, prefixes)):
        n = ql + pref
        ids = table[i, : math.ceil(n / BLOCK)].tolist()
        k = cache[ids, 0].reshape(-1, NKV, HD)[:n].float()
        v = cache[ids, 1].reshape(-1, NKV, HD)[:n].float()
        k = k.repeat_interleave(NQ // NKV, dim=1)
        v = v.repeat_interleave(NQ // NKV, dim=1)
        qq = q[off : off + ql].float()
        off += ql
        logits = torch.einsum("qhd,khd->hqk", qq, k) * scale
        qpos = torch.arange(pref, n, device=q.device)
        kpos = torch.arange(n, device=q.device)
        logits = logits.masked_fill(kpos[None, :] > qpos[:, None], float("-inf"))
        out.append(torch.einsum("hqk,khd->qhd", logits.softmax(-1), v).to(DTYPE))
    return torch.cat(out)


def reference_sdpa_early(q, cache, qs, prefixes, table):
    """Independent SDPA reference for small early-prefill audit shapes."""
    out = []
    off = 0
    for i, (ql, pref) in enumerate(zip(qs, prefixes)):
        n = ql + pref
        ids = table[i, : math.ceil(n / BLOCK)].tolist()
        k = cache[ids, 0].reshape(-1, NKV, HD)[:n].float()
        v = cache[ids, 1].reshape(-1, NKV, HD)[:n].float()
        mask = (
            torch.arange(n, device=q.device)[None, :]
            <= (pref + torch.arange(ql, device=q.device))[:, None]
        )
        qq = q[off : off + ql].float()
        off += ql
        output = F.scaled_dot_product_attention(
            qq.transpose(0, 1)[None],
            k.repeat_interleave(NQ // NKV, dim=1).transpose(0, 1)[None],
            v.repeat_interleave(NQ // NKV, dim=1).transpose(0, 1)[None],
            attn_mask=mask[None, None],
            dropout_p=0.0,
            is_causal=False,
        )
        out.append(output[0].transpose(0, 1).to(DTYPE))
    return torch.cat(out)


def kwargs(q, cache, cu, lens, table, out):
    return dict(
        q=q, k=cache[:, 0], v=cache[:, 1], out=out, cu_seqlens_q=cu,
        max_seqlen_q=0, seqused_k=lens, max_seqlen_k=int(lens.max()),
        softmax_scale=1.0 / math.sqrt(HD), causal=True, window_size=(-1, -1),
        block_table=table, softcap=0.0, q_descale=None, k_descale=None,
        v_descale=None, seq_threshold_3D=None, num_par_softmax_segments=None,
        softmax_segm_output=None, softmax_segm_max=None, softmax_segm_expsum=None,
        alibi_slopes=None, output_scale=None, qq_bias=None, sinks=None,
        mm_prefix_range=None, use_alibi_sqrt=False, kv_quant_mode=KVQuantMode.NONE,
        k_scale_cache=None, v_scale_cache=None, chunk_lookback=-1, use_td=False,
    )


def invoke(
    fn,
    kw,
    max_q,
    block_m=None,
    tile=None,
    warps=None,
    stages=None,
    pipeline_stages=None,
    scalar_block_lookup=False,
    mixed_dual_launch=False,
):
    kw = dict(kw, max_seqlen_q=max_q)
    if fn is tuned_attention:
        kw.update(
            block_m_override=block_m, tile_size_prefill_override=tile,
            launch_num_warps_override=warps, launch_num_stages_override=stages,
            pipeline_stages_override=pipeline_stages,
            scalar_block_lookup_override=bool(scalar_block_lookup),
            mixed_dual_launch_override=mixed_dual_launch,
        )
    fn(**kw)


def measure(fn, kw, max_q, warmup, iters, **overrides):
    for _ in range(warmup):
        invoke(fn, kw, max_q, **overrides)
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(iters):
        invoke(fn, kw, max_q, **overrides)
    ev1.record()
    ev1.synchronize()
    return ev0.elapsed_time(ev1) / iters


def run_case(device, name, qs, prefixes, cfg):
    q, cache, cu, lens, table = batch(device, qs, prefixes, cfg["seed"])
    expected = reference(q, cache, qs, prefixes, table)
    max_q = max(qs)
    base_out = torch.empty_like(q)
    tuned_out = torch.empty_like(q)
    base_kw = kwargs(q, cache, cu, lens, table, base_out)
    tuned_kw = kwargs(q, cache, cu, lens, table, tuned_out)
    invoke(unified_attention, base_kw, max_q)
    invoke(
        tuned_attention,
        tuned_kw,
        max_q,
        cfg["block_m"],
        cfg["tile"],
        cfg["warps"],
        cfg["stages"],
        cfg["pipeline_stages"],
        cfg["scalar_block_lookup"],
    )
    torch.cuda.synchronize()
    err_base = (base_out.float() - expected.float()).abs()
    err_tuned = (tuned_out.float() - expected.float()).abs()
    independent_error = None
    if name.startswith("early_"):
        independent = reference_sdpa_early(q, cache, qs, prefixes, table)
        independent_error = (independent.float() - expected.float()).abs().max().item()
    worst_index = tuple(int(i) for i in torch.unravel_index(err_base.argmax(), err_base.shape))
    base_ms = measure(unified_attention, base_kw, max_q, cfg["warmup"], cfg["iterations"])
    tuned_overrides = dict(
        block_m=cfg["block_m"], tile=cfg["tile"],
        warps=cfg["warps"], stages=cfg["stages"],
        pipeline_stages=cfg["pipeline_stages"],
        scalar_block_lookup=cfg["scalar_block_lookup"],
    )
    reverse = name.startswith("mixed") and cfg["mixed_dual_launch"] and cfg["dual_first"]
    tuned_ms = None if reverse else measure(
        tuned_attention, tuned_kw, max_q, cfg["warmup"], cfg["iterations"],
        **tuned_overrides,
    )
    dual_result = {}
    if name.startswith("mixed") and cfg["mixed_dual_launch"]:
        dual_out = torch.empty_like(q)
        dual_kw = kwargs(q, cache, cu, lens, table, dual_out)
        dual_overrides = dict(tuned_overrides, mixed_dual_launch=True)
        invoke(tuned_attention, dual_kw, max_q, **dual_overrides)
        torch.cuda.synchronize()
        dual_error = (dual_out.float() - expected.float()).abs()
        dual_ms = measure(
            tuned_attention, dual_kw, max_q,
            cfg["warmup"], cfg["iterations"], **dual_overrides
        )
        dual_result = dict(
            dual_ms=dual_ms,
            dual_max_abs=dual_error.max().item(),
            dual_over_005=int((dual_error > 0.05).sum().item()),
        )
    if reverse:
        tuned_ms = measure(
            tuned_attention, tuned_kw, max_q, cfg["warmup"], cfg["iterations"],
            **tuned_overrides,
        )
    if dual_result:
        dual_result["dual_vs_tuned"] = tuned_ms / dual_result["dual_ms"]
    return {
        "shape": name, "block_m": cfg["block_m"], "block_q": cfg["block_m"] // (NQ // NKV),
        "tile": cfg["tile"], "warps": cfg["warps"], "stages": cfg["stages"],
        "pipeline_stages": cfg["pipeline_stages"],
        "scalar_block_lookup": cfg["scalar_block_lookup"],
        "baseline_ms": base_ms, "tuned_ms": tuned_ms,
        "speedup": base_ms / tuned_ms, "baseline_tflops": flops(qs, prefixes, base_ms),
        "tuned_tflops": flops(qs, prefixes, tuned_ms),
        "baseline_max_abs": err_base.max().item(), "tuned_max_abs": err_tuned.max().item(),
        "baseline_mean_abs": err_base.mean().item(), "tuned_mean_abs": err_tuned.mean().item(),
        "baseline_worst_index": worst_index,
        "baseline_worst_reference": expected[worst_index].item(),
        "baseline_worst_actual": base_out[worst_index].item(),
        "tuned_at_baseline_worst": tuned_out[worst_index].item(),
        "baseline_over_005": int((err_base > 0.05).sum().item()),
        "tuned_over_005": int((err_tuned > 0.05).sum().item()),
        "reference_sdpa_max_abs": independent_error,
        **dual_result,
    }


def flops(qs, prefixes, ms):
    work = sum(2 * q * (p + q) * NQ * HD for q, p in zip(qs, prefixes))
    return work / (ms / 1000) / 1e12


def main():
    cfg = vars(args())
    device = torch.device(cfg.pop("device"))
    torch.cuda.set_device(device)
    shapes = []
    if cfg["shape"] in ("pure", "both"):
        shapes.append(("pure_prefill", [2048] * 8, [0, 2048, 4096, 6144, 8192, 10240, 12288, 14336]))
    if cfg["shape"] in ("mixed", "both"):
        shapes.append(("mixed", [2048] + [1] * 30, [8192] + [16384] * 30))
    if cfg["shape"] == "mixed-order":
        shapes.extend([
            ("mixed_decode_first", [1] * 30 + [2048], [16384] * 30 + [8192]),
            ("mixed_interleaved", [1] * 15 + [2048, 3] + [1] * 15,
             [16384] * 15 + [8192, 4096] + [16384] * 15),
        ])
    if cfg["shape"] == "early":
        shapes.extend([
            ("early_q16", [16], [0]),
            ("early_q16_prefix16", [16], [16]),
            ("early_q64", [64], [0]),
        ])
    results = []
    combos = [
        (m, t, w, s, ps, sb)
        for m in cfg["block_m"]
        for t in cfg["tile"]
        for w in cfg["warps"]
        for s in cfg["stages"]
        for ps in cfg["pipeline_stages"]
        for sb in cfg["scalar_block_lookup"]
    ]
    if cfg["limit"]:
        combos = combos[: cfg["limit"]]
    for name, qs, prefixes in shapes:
        for m, t, w, s, ps, sb in combos:
            # TILE_SIZE=8 is valid for the current 16-token KV page and is
            # required for the scalar-vs-vector page-table lookup ablation.
            if m % (NQ // NKV) or t < 8:
                continue
            c = dict(cfg, block_m=m, tile=t, warps=w, stages=s,
                     pipeline_stages=ps, scalar_block_lookup=sb)
            try:
                r = run_case(device, name, qs, prefixes, c)
            except Exception as exc:
                r = {"shape": name, "block_m": m, "tile": t, "warps": w, "stages": s,
                     "pipeline_stages": ps,
                     "scalar_block_lookup": sb,
                     "status": "error", "error": repr(exc)}
            else:
                r["status"] = (
                    "ok" if r["tuned_max_abs"] <= 0.05
                    and (r["reference_sdpa_max_abs"] is None
                         or r["reference_sdpa_max_abs"] <= 0.05)
                    and r.get("dual_max_abs", 0) <= 0.05
                    else "fail"
                )
            results.append(r)
            print(json.dumps(r, sort_keys=True), flush=True)
    cfg["results"] = results
    cfg["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    cfg["device"] = str(device)
    result_path = cfg["result"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    cfg["result"] = str(result_path)
    result_path.write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
