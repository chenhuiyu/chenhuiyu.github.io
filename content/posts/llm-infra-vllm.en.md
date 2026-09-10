---
title: "vLLM: from PagedAttention to measurable serving"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Infrastructure"
language: "en"
tags: ["llm-infra", "Hands-on", "Model Learning"]
pairKey: "llm-infra-vllm"
slug: "llm-infra-vllm-en"
excerpt: "Connect the scheduler, KV block manager, model runner and API server, then design a fair throughput experiment."
series: "llm-infra"
seriesOrder: 2
draft: false
---

![vLLM: from PagedAttention to measurable serving](/learning/llm-infra-vllm-en.svg)

## A framework is more than one kernel

PagedAttention is a foundational idea, but serving also requires request handling, lifecycle management, per-iteration scheduling, sampling and streaming. vLLM's architecture separates the frontend, scheduling core and device execution. This distinction helps identify slow CPU preprocessing, waiting for cache resources and slow GPU operators as different problems.

## Logical continuity, physical flexibility

A sequence consists of fixed-size logical blocks whose table entries point to physical storage. With block size 16, a 33-token sequence needs three blocks, using only one position in the last block. Paging bounds waste more finely rather than eliminating it; addressing and management have costs.

```python
lengths = [17, 33, 64]
block = 16
allocated = [((n+block-1)//block)*block for n in lengths]
print(allocated)  # [32, 48, 64]
print('slack tokens:', sum(allocated)-sum(lengths))  # 30
```

This capacity example assumes independent allocations without prefix sharing. Prefix reuse requires matching token sequences and relevant context, not merely semantically similar prompts.

## Establish a service, then change one variable

The basic single-GPU launch below requires a hardware-compatible vLLM installation. Record package version, model revision, dtype and driver. The model name is an example, not a performance recommendation.

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --host 127.0.0.1 --port 8000
```

First verify a short response and stopping behavior. Then fix the input/output length distribution and increase concurrency gradually. Separate cold and warm prefix-cache runs; replaying identical prompts can exaggerate expected production hit rates. Changing batching, quantization and model version together prevents attribution to paging.

## Check your understanding

**Question:** Is block size one necessarily cheapest?

<details><summary>Answer</summary>It reduces internal slack but can increase metadata, addressing, kernel and management costs. The best choice depends on implementation and request-length distribution, so measure it.</details>

## Primary sources and further reading

- [vLLM architecture](https://docs.vllm.ai/en/latest/design/arch_overview/)
- [vLLM quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/)
- [PagedAttention paper](https://arxiv.org/abs/2309.06180)

Sources checked on 2026-09-10. Teaching examples are not production benchmarks; confirm APIs and model support against the linked version.
