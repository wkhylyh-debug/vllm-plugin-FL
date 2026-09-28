"""Paged Split-KV attention for the Iluvatar vendor backend.

The kernel follows vLLM's flattened-query and paged-KV ABI. It is limited to
the unquantized causal path and is opt-in; unsupported paths remain on vLLM's
native unified attention implementation.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _split_kv_forward(
    block_batch_ptr,
    block_q_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    block_table_ptr,
    cu_q_ptr,
    seq_lens_ptr,
    o_split_ptr,
    m_split_ptr,
    l_split_ptr,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_bt,
    stride_os,
    stride_ot,
    stride_oh,
    stride_od,
    stride_ms,
    stride_mt,
    stride_mh,
    stride_ls,
    stride_lt,
    stride_lh,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    num_splits: tl.constexpr,
    max_kv_blocks_per_split: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    scale: tl.constexpr,
):
    linear_block = tl.program_id(0)
    q_head = tl.program_id(1)
    batch = tl.load(block_batch_ptr + linear_block)
    q_block = tl.load(block_q_ptr + linear_block)
    split = tl.program_id(2)

    q_begin = tl.load(cu_q_ptr + batch)
    q_end = tl.load(cu_q_ptr + batch + 1)
    q_len = q_end - q_begin
    context_len = tl.load(seq_lens_ptr + batch)
    split_size = (context_len + num_splits - 1) // num_splits
    split_begin = split * split_size
    split_end = tl.minimum(split_begin + split_size, context_len)

    q_local = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    q_valid = q_local < q_len
    q_abs = context_len - q_len + q_local
    q_global = q_begin + q_local
    d = tl.arange(0, BLOCK_D)
    q_ptrs = (
        q_ptr
        + q_global[:, None] * stride_qt
        + q_head * stride_qh
        + d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=q_valid[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    gqa = num_q_heads // num_kv_heads
    kv_head = q_head // gqa

    for tile_idx in range(max_kv_blocks_per_split):
        kv_pos = split_begin + tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_valid = kv_pos < split_end
        physical_block = kv_pos // block_size
        offset = kv_pos - physical_block * block_size
        block_number = tl.load(
            block_table_ptr + batch * stride_bt + physical_block,
            mask=kv_valid,
            other=0,
        )

        k_ptrs = (
            k_ptr
            + block_number[None, :] * stride_kb
            + offset[None, :] * stride_kt
            + kv_head * stride_kh
            + d[:, None] * stride_kd
        )
        v_ptrs = (
            v_ptr
            + block_number[:, None] * stride_vb
            + offset[:, None] * stride_vt
            + kv_head * stride_vh
            + d[None, :] * stride_vd
        )
        k = tl.load(k_ptrs, mask=kv_valid[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=kv_valid[:, None], other=0.0)

        qk = tl.dot(q, k) * scale
        causal = q_abs[:, None] >= kv_pos[None, :]
        valid = q_valid[:, None] & kv_valid[None, :] & causal
        qk = tl.where(valid, qk, -float("inf"))

        m_ij = tl.max(qk, axis=1)
        has_keys = tl.sum(valid.to(tl.int32), axis=1) > 0
        m_new = tl.where(has_keys, tl.maximum(m_i, m_ij), m_i)
        safe_max = tl.where(has_keys, m_new, 0.0)
        alpha_old = tl.where(
            has_keys,
            tl.where(l_i > 0, tl.exp(m_i - safe_max), 0.0),
            1.0,
        )
        p = tl.exp(tl.where(valid, qk - safe_max[:, None], -float("inf")))
        l_new = alpha_old * l_i + tl.sum(p, axis=1)
        acc = acc * alpha_old[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
        l_i = l_new

    o_ptrs = (
        o_split_ptr
        + split * stride_os
        + q_global[:, None] * stride_ot
        + q_head * stride_oh
        + d[None, :] * stride_od
    )
    m_ptr = m_split_ptr + split * stride_ms + q_global * stride_mt + q_head * stride_mh
    l_ptr = l_split_ptr + split * stride_ls + q_global * stride_lt + q_head * stride_lh
    tl.store(o_ptrs, acc / tl.maximum(l_i[:, None], 1e-20), mask=q_valid[:, None])
    tl.store(m_ptr, m_i, mask=q_valid)
    tl.store(l_ptr, l_i, mask=q_valid)


@triton.jit
def _split_kv_reduce(
    block_batch_ptr,
    block_q_ptr,
    o_split_ptr,
    m_split_ptr,
    l_split_ptr,
    out_ptr,
    cu_q_ptr,
    stride_os,
    stride_ot,
    stride_oh,
    stride_od,
    stride_ms,
    stride_mt,
    stride_mh,
    stride_ls,
    stride_lt,
    stride_lh,
    stride_ot_out,
    stride_oh_out,
    stride_od_out,
    num_q_heads: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    linear_block = tl.program_id(0)
    q_head = tl.program_id(1)
    batch = tl.load(block_batch_ptr + linear_block)
    q_block = tl.load(block_q_ptr + linear_block)
    q_begin = tl.load(cu_q_ptr + batch)
    q_end = tl.load(cu_q_ptr + batch + 1)
    q_len = q_end - q_begin
    q_local = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    q_valid = q_local < q_len
    q_global = q_begin + q_local
    d = tl.arange(0, BLOCK_D)

    m_global = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    for split in range(num_splits):
        m_ptr = m_split_ptr + split * stride_ms + q_global * stride_mt + q_head * stride_mh
        m_global = tl.maximum(m_global, tl.load(m_ptr, mask=q_valid, other=-float("inf")))

    denom = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    for split in range(num_splits):
        m_ptr = m_split_ptr + split * stride_ms + q_global * stride_mt + q_head * stride_mh
        l_ptr = l_split_ptr + split * stride_ls + q_global * stride_lt + q_head * stride_lh
        m_i = tl.load(m_ptr, mask=q_valid, other=-float("inf"))
        l_i = tl.load(l_ptr, mask=q_valid, other=0.0)
        active = q_valid & (l_i > 0)
        weight = tl.where(active, tl.exp(m_i - m_global) * l_i, 0.0)
        o_ptrs = (
            o_split_ptr
            + split * stride_os
            + q_global[:, None] * stride_ot
            + q_head * stride_oh
            + d[None, :] * stride_od
        )
        o_i = tl.load(o_ptrs, mask=q_valid[:, None], other=0.0).to(tl.float32)
        denom += weight
        acc += weight[:, None] * o_i

    out_ptrs = (
        out_ptr
        + q_global[:, None] * stride_ot_out
        + q_head * stride_oh_out
        + d[None, :] * stride_od_out
    )
    tl.store(out_ptrs, acc / tl.maximum(denom[:, None], 1e-20), mask=q_valid[:, None])


def paged_split_kv_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    num_splits: int = 4,
    block_m: int = 64,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Run Split-KV on the exact flattened/paged vLLM attention ABI."""
    if q.ndim != 3 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("expected q=[T,H,D] and k/v=[blocks,block,kv_heads,D]")
    if q.device != k.device or q.device != v.device or q.device != out.device:
        raise ValueError("q, k, v, and out must be on the same device")
    if out.shape != q.shape or k.shape != v.shape:
        raise ValueError("out must match q and k/v must have identical shapes")
    if num_splits not in (2, 4, 8, 16):
        raise ValueError("num_splits must be one of 2, 4, 8, 16")
    if block_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("block_table must contain integer physical block ids")
    if cu_seqlens_q.ndim != 1 or seqused_k.ndim != 1:
        raise ValueError("cu_seqlens_q and seqused_k must be one-dimensional")
    if cu_seqlens_q.numel() != seqused_k.numel() + 1:
        raise ValueError("cu_seqlens_q must contain one entry per request plus one")
    if block_table.shape[0] != seqused_k.numel():
        raise ValueError("block_table and seqused_k must have the same request count")

    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    block_size = k.shape[1]
    if (
        head_dim not in (64, 128, 256)
        or num_q_heads % num_kv_heads
        or block_size <= 0
        or block_size % 16
        or block_m <= 0
        or block_m % 16
        or block_n <= 0
        or block_n % 16
    ):
        raise ValueError("unsupported head shape for paged Split-KV")
    if max_seqlen_q <= 0 or max_seqlen_k <= 0:
        return out.zero_()

    q_blocks_per_request = (
        (cu_seqlens_q[1:] - cu_seqlens_q[:-1] + block_m - 1) // block_m
    )
    total_q_blocks = int(q_blocks_per_request.sum().item())
    request_ids = torch.arange(
        block_table.shape[0], device=q.device, dtype=torch.int32
    )
    block_batch = torch.repeat_interleave(request_ids, q_blocks_per_request)
    block_starts = torch.cumsum(q_blocks_per_request, dim=0) - q_blocks_per_request
    block_q = torch.arange(total_q_blocks, device=q.device, dtype=torch.int32)
    block_q = block_q - torch.repeat_interleave(block_starts, q_blocks_per_request)
    max_split_blocks = math.ceil(max_seqlen_k / num_splits / block_n) + 1
    o_splits = torch.empty(
        (num_splits, total_q, num_q_heads, head_dim),
        device=q.device,
        dtype=q.dtype,
    )
    m_splits = torch.full(
        (num_splits, total_q, num_q_heads),
        -float("inf"),
        device=q.device,
        dtype=torch.float32,
    )
    l_splits = torch.zeros_like(m_splits)

    grid = (total_q_blocks, num_q_heads, num_splits)
    _split_kv_forward[grid](
        block_batch,
        block_q,
        q,
        k,
        v,
        block_table,
        cu_seqlens_q,
        seqused_k,
        o_splits,
        m_splits,
        l_splits,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        block_table.stride(0),
        o_splits.stride(0), o_splits.stride(1), o_splits.stride(2), o_splits.stride(3),
        m_splits.stride(0), m_splits.stride(1), m_splits.stride(2),
        l_splits.stride(0), l_splits.stride(1), l_splits.stride(2),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        num_splits=num_splits,
        max_kv_blocks_per_split=max_split_blocks,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=head_dim,
        scale=softmax_scale,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    reduce_grid = (total_q_blocks, num_q_heads)
    _split_kv_reduce[reduce_grid](
        block_batch,
        block_q,
        o_splits,
        m_splits,
        l_splits,
        out,
        cu_seqlens_q,
        o_splits.stride(0), o_splits.stride(1), o_splits.stride(2), o_splits.stride(3),
        m_splits.stride(0), m_splits.stride(1), m_splits.stride(2),
        l_splits.stride(0), l_splits.stride(1), l_splits.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        num_q_heads=num_q_heads,
        num_splits=num_splits,
        BLOCK_M=block_m,
        BLOCK_D=head_dim,
        num_warps=num_warps,
    )
    return out
