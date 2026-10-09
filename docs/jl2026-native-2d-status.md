# JL2026 Native 2D Candidate Status

Stage checkpoint: October 9, 2026. The old Iluvatar machine is unavailable;
the next hardware environment has not been validated.

## Repository Boundary

This branch is the formal candidate implementation, not the experiment
workspace. Its production code baseline is commit
`02badf74e8a10aabc6f401407608c097ca30b163`.

The stage-close change only removes standalone experimental archives and
adds this status note. It does not change operator source, dispatch behavior,
the official benchmark, or the focused correctness harness.

Experiment scripts, rejected designs, the full experiment record, downloaded
raw results, and unvalidated Decode patches are maintained in
[the experiment repository](https://github.com/AlexMa616/minicpm-attention-optimization-iluvatar).
Start with its
[stage summary](https://github.com/AlexMa616/minicpm-attention-optimization-iluvatar/blob/main/docs/stage-summary-2026-10-09.md).
Do not apply archived patches or run legacy restart scripts automatically.

## Candidate Implementation

- `vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention.py`: guarded,
  opt-in dispatch; unsupported features and decode fall back upstream.
- `vllm_fl/dispatch/backends/vendor/iluvatar/impl/ops/triton_unified_attention_native.py`:
  tiled online-softmax attention with FP32 accumulation and paged-KV/GQA support.
- Frozen prefill configuration: `BLOCK_M=128`, `TILE_SIZE=16`, `num_warps=8`,
  launch `num_stages=2`, loop `pipeline_stages=1`.
- `tools/native_2d_harness.py`: focused correctness and kernel timing.
- `tools/start_native_2d_service.sh`: candidate service launcher.

Experimental scalar lookup and dual-launch parameters still exist in the
kernel interface, but the formal dispatcher does not enable them. Their
presence is not an acceptance claim. This checkpoint does not refactor that
interface without device regression tests.

## Evidence Limits

Historical BF16 candidate Level 3 completed 105 questions with 102 correct
(`97.14%`). That result does not cover unvalidated patches or a new machine.
Large prefill-heavy short-benchmark gains are not final official gains.
The subsequent Decode experiment is not merged here: kernel-only gains
did not establish stable service-level gains.

Latest isolated routing evidence on the old machine loaded the Iluvatar
vendor backend but observed Native guard fallback. Source inspection and a
synthetic test point to non-None `k_descale/v_descale` in the BF16/auto caller;
the final real per-guard smoke result was not recovered. Do not describe the
candidate as guaranteed to route Native on a fresh environment.

Next: validate the new environment, confirm actual guards and scale semantics,
test any isolated fix against upstream, re-capture graphs, and only then run
comparable service A/B and full accuracy acceptance. No unvalidated source
fix is included in this stage-close commit.
