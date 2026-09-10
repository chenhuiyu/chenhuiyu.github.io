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

## A complete explanation from first principles

### Why naive KV allocation wastes memory

At request admission, the service often does not know the final output length. Reserving a large contiguous buffer leaves unused space for short answers. Repeatedly growing contiguous storage can introduce movement or fragmentation. Limited memory restricts concurrency and therefore batching opportunities.

PagedAttention separates a sequence's logical order from its physical KV storage. Logical blocks can occupy nonadjacent physical locations, with a block table maintaining the mapping. The attention kernel follows that mapping while preserving the context's logical order.

### Calculate a block-allocation example

If a block holds sixteen tokens, a 35-token sequence needs three blocks: capacity 48, with thirteen unused slots. A 17-token sequence needs two blocks. Paging does not remove all waste; it limits tail waste to block granularity and makes allocation more flexible.

```python
block = 16
for length in [35,17,64]:
    blocks = (length+block-1)//block
    print(length, blocks, 'unused slots:', blocks*block-length)
```

This calculates slots only. It neither measures a kernel nor models prefix sharing. Smaller blocks can reduce tail waste while increasing metadata and management costs, so this arithmetic alone cannot identify the best block size.

### Continuous batching changes when membership changes

A static batch may wait for all its requests to complete before replacing them. If one answer requires ten tokens and another five hundred, the short request's capacity may remain underused. Continuous batching updates the active request set between iterations, removing completed requests and admitting new work.

Queueing does not disappear. The scheduler remains constrained by token budgets, KV capacity and policy. Understand memory management and scheduling together rather than reducing the framework to a single kernel name.

### Prefix reuse requires compatible computation

Requests with an identical prefix may reuse its computed states when the relevant conditions match. Semantic similarity is insufficient: different token sequences generally cannot share the same KV. Model weights, position handling and other computation-affecting conditions also need compatibility.

A changing timestamp near the beginning of a prompt can break a large common prefix. Improving prompt layout must preserve meaning and isolation; increased cache reuse does not justify combining private user content incorrectly.

### Build a comparison that explains its result

After startup, send a fixed short request and verify tokenization, templates, precision and output behavior. Then test length and concurrency. Record framework version, model revision, hardware, parallelism and request distributions. Match sampling and stopping rules across frameworks; higher tokens per second could otherwise reflect different output lengths or quality.

Separate cold startup, steady operation without reuse and highly shared-prefix workloads. Cache-hit speed is not a universal request speed. If performance falls unexpectedly, inspect length distributions, memory pressure and preemption or recomputation before moving to low-level traces.

The browser notebook verifies memory accounting and parallel-matrix algebra; it does not run a complete GPU vLLM server. Use the official deployment material for hardware experiments. Predicting memory behavior and checking it against deployment metrics transfers better than memorizing a launch command.

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
