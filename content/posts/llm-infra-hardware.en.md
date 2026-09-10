---
title: "GPU, TPU and kernels: why FLOPs do not explain speed"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Infrastructure"
language: "en"
tags: ["llm-infra", "Hands-on", "Model Learning"]
pairKey: "llm-infra-hardware"
slug: "llm-infra-hardware-en"
excerpt: "Use Roofline to understand HBM, on-chip storage, matrix units, tiling and fusion, then locate Triton and Pallas in the stack."
series: "llm-infra"
seriesOrder: 7
draft: false
---

![GPU, TPU and kernels: why FLOPs do not explain speed](/learning/llm-infra-hardware-en.svg)

## Follow data through memory

GPU execution involves threads/warps, SM resources, registers, shared memory and HBM. Tensor Cores accelerate suitable matrix operations rather than replacing all general computation. TPUs combine matrix, vector/scalar resources and on-chip memory under compiler-driven execution. Generations differ; “GPUs are flexible, TPUs only multiply matrices” is not an adequate model.

Data movement is often expensive. Writing every intermediate to HBM and rereading it in another kernel can dominate modest FLOP counts. Tiling reuses loaded data in faster storage. Fusion reduces intermediate traffic but may increase register pressure or reduce parallelism.

## Two Roofline bounds

Arithmetic intensity is $I=\mathrm{FLOPs}/\mathrm{bytes}$. A simplified throughput bound is $\min(P_{peak},BW\times I)$, with time lower bound $\max(F/P_{peak},M/BW)$. This is an idealized bound, not a latency predictor. Launches, synchronization, imbalance and communication add costs.

```python
flops = 2 * 1024**3
bytes_moved = 3 * 1024**2 * 2  # Teaching assumption: each matrix transferred once.
intensity = flops / bytes_moved
print('FLOP/byte:', intensity)
```

Real GEMMs have cache reuse, layouts and dtype effects; this three-matrix estimate is not universally accurate traffic accounting.

## Establish shapes before writing a kernel

Triton's tiled matmul tutorial introduces program instances and blocked, masked memory operations. JAX/Pallas offers lower-level kernel programming for supported TPU paths, with backend-specific constraints. Optimize in order: numerical correctness, safe tails, correct strides/layouts, then block sizes, warps and pipeline stages.

Compare random shapes and nondivisible dimensions against a reference before benchmarking. A kernel fast for 1024×1024 may lose on thin decode matrices. State whether a multiply-add counts as two FLOPs, whether transfers are timed and which precision is used.

## Check your understanding

**Question:** A kernel reaches only 10% of peak compute. Is it necessarily poorly written?

<details><summary>Answer</summary>No. It may be near the bandwidth bound at low arithmetic intensity. Compare the relevant Roofline first, then inspect memory efficiency, occupancy and synchronization. High occupancy is not the final objective either.</details>

## Primary sources and further reading

- [TPU architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)
- [Triton matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Pallas](https://docs.jax.dev/en/latest/pallas/index.html)

Sources checked on 2026-09-10. Teaching examples are not production benchmarks; confirm APIs and model support against the linked version.
