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

## A complete explanation from first principles

### Accelerators need both arithmetic and a supply of data

CPUs handle complex control and low-latency serial work well. GPUs devote many execution resources to regular parallel computation. Registers, on-chip shared storage and device memory have different capacities and access costs. Data reuse can matter as much as operation count.

TPUs also specialize in tensor computation, but their compilation, execution and memory organization should not be reduced to CUDA's thread model. Common questions are how matrix units receive data, how on-chip capacity limits tiling and how devices communicate. Consult vendor material for generation-specific specifications.

### Use arithmetic intensity to choose a direction

Arithmetic intensity measures FLOPs per transferred byte. A simplified roofline bound is `min(peak compute, memory bandwidth*arithmetic intensity)`. For a hypothetical 100-TFLOP/s device with effective bandwidth one TB/s, a kernel at ten FLOPs/byte has a bandwidth ceiling of ten TFLOP/s.

These are illustrative values, not measured hardware results. Match precision and sparsity assumptions between operation counts and peaks. Launch overhead, dependencies and communication can keep execution well below even this simplified bound.

### Tiling increases reuse within limited storage

In matrix multiplication, an element of A participates in several products with B. Repeatedly fetching it from external memory wastes bandwidth. A tile holds nearby pieces in faster storage and reuses them.

Larger tiles can exhaust registers or shared memory, reduce concurrent residency or cause spills. Optimization balances reuse and resource occupancy. High occupancy by itself does not guarantee efficient accesses or high performance.

### Fusion can eliminate intermediate traffic

Separate bias, activation and scaling kernels may repeatedly write and read intermediate tensors. A fused kernel can keep local values within one execution, reducing traffic and launch count.

Fusion can also increase register pressure or prevent use of a well-optimized library operation. Compare the complete subgraph, including numerical correctness, rather than declaring victory from a lower kernel count.

### Input precision and accumulation precision are different

Low-precision inputs do not require equally low-precision accumulation. Long reductions can magnify numerical error, so implementations may use distinct input, accumulator and output formats. BF16 and FP16 occupy two bytes but allocate exponent range and precision differently.

Validate small shapes, boundary sizes, non-divisible dimensions and extreme values against a reference before timing. Set tolerances appropriate to the datatype and task, then warm up and synchronize appropriately. A fast incorrect kernel is not an optimization.

### Build a reference-to-measurement learning loop

Start with correct vector operations, reductions and matrix multiplication in a high-level implementation. Study how access patterns, tiles and fusion change execution. GPU work benefits from both system traces and kernel counters; TPU work also requires understanding compiled graphs, shapes, layouts and sharding.

The notebook's Python loops illustrate operation growth and timing noise, not GPU-kernel performance. Doubling a square matrix side multiplies ordinary cubic work by about eight; observed time also reflects interpreter and device conditions. Explaining that gap is part of learning performance engineering.

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
