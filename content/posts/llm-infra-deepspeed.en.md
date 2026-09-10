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
