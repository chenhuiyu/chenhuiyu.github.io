---
title: "DeepSpeed ZeRO: account for every training state"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Infrastructure"
language: "en"
tags: ["llm-infra", "Hands-on", "Model Learning"]
pairKey: "llm-infra-deepspeed"
slug: "llm-infra-deepspeed-en"
excerpt: "Start with Adam weights, gradients and optimizer state; derive ZeRO-1/2/3 memory savings and communication costs."
series: "llm-infra"
seriesOrder: 6
draft: false
---

![DeepSpeed ZeRO: account for every training state](/learning/llm-infra-deepspeed-en.svg)

## A complete explanation from first principles

### Training stores more than weights

One common mixed-precision Adam ledger uses two bytes for low-precision weights, two for gradients, four for master weights and eight for two moments: sixteen bytes per parameter. Implementations differ, so this is an explicitly chosen accounting convention rather than a universal constant.

For seven billion parameters, persistent training state is approximately 112 GB, or 104.3 GiB, before activations. Selecting a training device from the fourteen-GB weight footprint alone misses most of the state.

### ZeRO removes replicated state in stages

Ordinary data-parallel ranks commonly replicate optimizer state, gradients and parameters. ZeRO-1 partitions optimizer state; ZeRO-2 also partitions gradients; ZeRO-3 additionally partitions parameters. Logical data-parallel training remains, while long-term ownership changes.

Under the ledger above and N ranks, per-rank estimates are `4P+12P/N` bytes for Stage 1, `2P+14P/N` for Stage 2 and `16P/N` for Stage 3. These exclude activations, transient gathered parameters, communication buffers and allocator effects. Stage 3's persistent estimate is not a peak-memory guarantee.

### Partitioned parameters still have to participate in computation

A device cannot compute a full layer independently from an arbitrary fraction of its weights. Required parameter pieces must be gathered or otherwise made available according to the execution schedule, then potentially released. Backward communication establishes the appropriate gradient shards. Memory savings trade against communication and scheduling work.

A very large individual layer can still create a temporary gathering peak. Prefetch can hide transfer time while consuming additional memory. Tune against measured peaks and timelines instead of assuming more aggressive prefetch is always better.

### Activation checkpointing addresses a different ledger

Backward computation needs intermediate forward information. Checkpointing saves fewer activations and recomputes some during backward, exchanging computation for memory. It does not partition optimizer state. With long sequences, activations can dominate even after ZeRO has substantially reduced persistent state.

If parameter state falls from sixty GiB to ten while activations remain twenty, activations become the main component. A moving bottleneck is an expected consequence of successful optimization.

### Offload trades capacity against movement

CPU or NVMe offload reduces GPU residency but may add transfer and CPU-update costs. Host-memory capacity, PCIe bandwidth, NUMA placement and storage behavior matter. Making a larger model runnable and accelerating the same model are separate goals.

Decide whether capacity or throughput is limiting. Offload may be worthwhile when the alternative is an impossible configuration, yet unnecessary when the model already fits comfortably on the accelerator.

Use the notebook to vary parameter count, rank count and stage. Reserve separate budgets for activations and temporary buffers, then compare with real peak allocated and reserved memory using the implementation's metric definitions. Finally verify losses and updates against a baseline. Incorrect token normalization, accumulation or learning-rate handling can create a system that fits but optimizes a different objective.

## Why parameter count is not training memory

Consider one mixed-precision Adam configuration: two bytes for low-precision weights, two for gradients, four for FP32 master weights and eight for two FP32 moments. That totals 16 bytes per parameter. Not every BF16 setup keeps master weights or two-byte gradients; state the implementation before calculating.

Under this assumption, 7B parameters require about 104.3 GiB of persistent training state before activations, attention workspace and communication buffers. Approximately 14 GB of inference weights does not imply the model can be fully trained in that memory.

## What each ZeRO stage partitions

| Stage | Partitioned across data-parallel ranks | Still replicated locally |
|---|---|---|
| ZeRO-1 | Optimizer state, including master weights here | Weights and gradients |
| ZeRO-2 | Also gradients | Weights |
| ZeRO-3 | Also parameters | Temporary gathered computation state |

With P ranks, this example's persistent bytes per rank change from $16N$ to $4N+12N/P$, then $2N+14N/P$, and approximately $16N/P$. Edit N and P in the Python experiment. Peak memory can be substantially higher because current-layer parameters are gathered for computation.

## Different from TP and recomputation

ZeRO reduces redundancy among data-parallel replicas; TP parallelizes individual operators. They can be combined. Activation checkpointing saves fewer forward activations and recomputes during backward, trading compute for memory. CPU/NVMe offload moves storage to slower tiers and adds transfers; capacity improvement is not equivalent to throughput improvement.

Stage three is not the end of tuning. Inspect whether parameter gathers and gradient reduce-scatter overlap computation. Tiny microbatches may expose communication overhead. A single oversized layer may still require tensor partitioning or other techniques.

## Check your understanding

**Question:** If eight-rank ZeRO-3 divides the persistent-state lower bound by eight, must peak memory also divide exactly by eight?

<details><summary>Answer</summary>No. Activations, temporary gathers, communication buffers and unsharded objects may not scale that way. Measure peak memory and the largest layer, not only static formulas.</details>

## Primary sources and further reading

- [DeepSpeed ZeRO tutorial](https://www.deepspeed.ai/tutorials/zero/)
- [ZeRO paper](https://arxiv.org/abs/1910.02054)

Sources checked on 2026-09-10. Teaching examples are not production benchmarks; confirm APIs and model support against the linked version.
