---
title: "XProf / Nsight: find the bottleneck in a timeline"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Infrastructure"
language: "en"
tags: ["llm-infra", "Hands-on", "Model Learning"]
pairKey: "llm-infra-profiling"
slug: "llm-infra-profiling-en"
excerpt: "Inspect end-to-end timelines before slow kernels; use warm-up, synchronization and controlled experiments to avoid timing traps."
series: "llm-infra"
seriesOrder: 8
draft: false
---

![XProf / Nsight: find the bottleneck in a timeline](/learning/llm-infra-profiling-en.svg)

## First ask why the accelerator is idle

Nsight Systems examines CPU, CUDA and communication timelines. Nsight Compute investigates individual kernels. XProf provides accelerator analysis and traces, especially useful in XLA/JAX/TPU workflows. They operate at different levels. A low-utilization kernel alone cannot explain end-to-end latency.

Capture a representative steady-state window covering loading, tokenization, CPU dispatch, GEMMs, attention, collectives and synchronization. Gaps may indicate waiting, Python overhead, compilation or networking rather than insufficient device compute.

## Correct timing precedes tuning

GPU operations are usually asynchronous: Python returning does not mean a kernel finished. Use CUDA events or synchronized wall time, and warm up to separate compilation, initialization and cold caches. Do not call CUDA synchronization for CPU-only execution. The notebook selects timing by device and records distributions and environment.

```bash
# Requires CUDA/Nsight and your prepared bench.py.
nsys profile --trace=cuda,nvtx,osrt -o baseline python bench.py
ncu --set basic --target-processes all python bench.py
```

Profiling adds overhead; detailed kernel collection may replay execution. Separate profiling windows from normal performance runs instead of presenting instrumented time as user latency.

## Turn observations into falsifiable hypotheses

CPU gaps between decode steps suggest investigating dispatch or compatible graph execution. A long NCCL tail suggests buckets, overlap or a slow rank. Near-saturated effective HBM bandwidth with low compute utilization suggests reuse, precision or cache traffic. Change one major factor and remeasure end-to-end quality and latency.

For TPU/JAX, account for compilation, asynchronous dispatch and `block_until_ready()`. Compiled HLO/fusion boundaries need not correspond one-to-one with framework operators; map them back to source computations.

## Check your understanding

**Question:** Detailed profiling makes the model five times slower. Is that a regression?

<details><summary>Answer</summary>Not enough evidence. Exclude collection overhead, replay and altered synchronization. Establish actual performance with a stable uninstrumented benchmark, then use traces to explain it.</details>

## Actual output from this run

![CPU matmul timings measured during authoring, p50/p95 over 20 repeats. These are not GPU or TPU benchmarks.](/learning/llm-infra-profiling-actual.png)

CPU matmul timings measured during authoring, p50/p95 over 20 repeats. These are not GPU or TPU benchmarks.

## Primary sources and further reading

- [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [Nsight Compute profiling](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [XProf](https://openxla.org/xprof)
- [JAX asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html)

Sources checked on 2026-09-10. Teaching examples are not production benchmarks; confirm APIs and model support against the linked version.
