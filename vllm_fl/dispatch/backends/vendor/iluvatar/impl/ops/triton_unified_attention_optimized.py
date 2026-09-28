"""Compatibility entry point for the validated Iluvatar Split-KV kernel."""

from __future__ import annotations

import os
from typing import Optional

import torch

from .triton_split_kv_paged import paged_split_kv_attention


def unified_attention_optimized(
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
    causal: bool = True,
    num_splits_override: Optional[int] = None,
    block_m_override: Optional[int] = None,
    block_n_override: Optional[int] = None,
    launch_num_warps_override: Optional[int] = None,
    launch_num_stages_override: Optional[int] = None,
    **kwargs,
) -> torch.Tensor:
    if causal is not True:
        raise ValueError("Split-KV currently supports causal attention only")
    return paged_split_kv_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        block_table=block_table,
        seqused_k=seqused_k,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        num_splits=num_splits_override
        or int(os.environ.get("ILUVATAR_NUM_SPLITS", "0"))
        or int(os.environ.get("VLLM_ILUVATAR_ATTN_SPLIT_KV_SEGMENTS", "4")),
        block_m=block_m_override or 64,
        block_n=block_n_override or 64,
        num_warps=launch_num_warps_override or 4,
        num_stages=launch_num_stages_override or 2,
    )
