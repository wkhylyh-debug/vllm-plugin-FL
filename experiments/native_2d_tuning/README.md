# Native 2D Attention Experiments

This directory contains diagnostic and rejected alternatives from the
MiniCPM5-2B Iluvatar optimization work. Nothing here is imported by the
production dispatch path.

## Production result

The submitted path is the Native 2D kernel in
`vllm_fl/dispatch/backends/vendor/iluvatar/impl/ops/triton_unified_attention_native.py`.
The production dispatcher keeps decode and unsupported attention features on
the upstream vLLM implementation.

The measured candidate configuration is:

- `BLOCK_M=128`
- `TILE_SIZE=16`
- `num_warps=8`
- `num_stages=2`
- `pipeline_stages=1`

## Rejected alternatives

`rejected_split_kv/` contains the Split-KV prototype. It was removed from the
production source tree after service-level tests showed no stable gain.

`service_and_profiler/` contains A/B, profiling, dual-launch, and parallel
scan scripts used during development. They are retained only for auditability
and are not required to run the final service.

The official benchmark harness remains under `benchmarks/`, and the focused
kernel harness remains under `tools/native_2d_harness.py`.
