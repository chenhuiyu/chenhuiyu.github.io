---
title: "SGLang: shared prefixes, structured generation and PD separation"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Infrastructure"
language: "en"
tags: ["llm-infra", "Hands-on", "Model Learning"]
pairKey: "llm-infra-sglang"
slug: "llm-infra-sglang-en"
excerpt: "Understand RadixAttention through repeated prefixes, then budget the cost of transferring KV between workers."
series: "llm-infra"
seriesOrder: 3
draft: false
---

![SGLang: shared prefixes, structured generation and PD separation](/learning/llm-infra-sglang-en.svg)

## A complete explanation from first principles

### Repeated prefixes appear naturally in real workflows

Applications often contain conversations under a shared system prompt, multiple questions about one document, or several candidate completions for a task. Recomputing every common prefix wastes prefill work. One useful way to understand SGLang is through organization and reuse of those prefixes.

A radix tree is a compressed prefix tree: common beginnings share a path, which branches where inputs diverge. The reusable object is the computation state of a matching token prefix, not a copied answer to a semantically similar question.

### Work through a branching example

Consider `[system, document A, question 1]`, `[system, document A, question 2]`, and `[system, document B, question 3]`. The first two share a longer prefix; the third generally shares only the system section. Putting the question before the document makes branching occur earlier, so an identical later document is not necessarily reusable as a prefix.

Prompt layout therefore affects performance, but task semantics come first. Reordering evidence can change answers. Cache optimization must preserve quality, and finite memory means cached prefixes eventually need eviction.

### Structured generation constrains form

A decoder can restrict token choices to those compatible with a grammar or schema. This more directly controls output form than merely requesting JSON in the prompt. Yet `{"price":999}` can be perfectly valid JSON and still contain a wrong price.

Types, enumerations, required fields, business rules and factual grounding are distinct validation layers. Complex constraints can also introduce preparation and execution costs. Measure those costs within the actual request path, including differences between first use and cached operation.

### Disaggregating prefill and decode introduces a transfer

Because prefill and decode have different execution patterns, separate resource pools can reduce interference and allow independent capacity tuning. The handoff requires transferring KV state, coordinating processes and balancing the two pools.

For a 512 MiB cache over an effective 50 GiB/s link, data transfer alone has a lower bound near ten milliseconds, before protocol, queueing and synchronization costs. This is an idealized calculation, not a deployment measurement. Short requests may not amortize the added overhead.

### Construct an informative experiment

Test unshared prompts, highly shared fixed prefixes, and shared prefixes with changing suffixes. Control input lengths and output budgets. Record cache hit rate, TTFT, ITL, throughput and cache occupancy. For disaggregation, also measure KV transfer time and queueing on both sides to determine whether capacity is genuinely balanced.

For structured outputs, report schema compliance, semantic correctness and latency separately. Improved formatting does not establish increased intelligence. Verify meaning on a small sample before scaling the load.

### Compare implementations against your workload

Serving frameworks have overlapping and evolving capabilities. Avoid permanently assigning each feature to one project. Ask whether the model and hardware are supported, whether the workflow is covered, how tail latency behaves, what observability exists and how expensive upgrades are. Use official documentation to verify current interfaces and the prefix, constraint and phase models here to explain measurements.

## Why repeated beginnings matter

Multi-turn conversations, few-shot tasks and tool workflows often share beginnings. Matching tokens, model state and positions can permit prefix K/V reuse. A radix tree compresses common prefixes and organizes branches; this is a different abstraction from paged physical storage. The goal is exact computation reuse, not semantic clustering.

Requests `[A,B,C,X]` and `[A,B,C,Y]` share three tokens. Reuse may save their prefill work, but the fourth token and subsequent decoding still require computation. A changing timestamp at the beginning can prevent a later common passage from matching as a prefix.

## Structured output has limits

Constrained decoding restricts allowable tokens to satisfy syntax or a schema. Valid JSON does not establish factual fields, and a legal tool name does not establish a sensible call. Measure syntax validity, schema validity, semantic correctness and latency separately. Constraints alter the sampling distribution and can affect quality.

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 --port 30000
```

Check installation and device support in the official documentation. Compare frameworks using matched models and requests, not unrelated default configurations.

## PD separation is not free

A prefill worker sends KV to a decode worker. A transfer lower bound is bytes divided by effective bandwidth, before scheduling, protocol, layout and synchronization costs. Under a teaching assumption of 2 GiB and 25 GiB/s effective bandwidth, the lower bound is 80 ms. This is not measured latency and must not substitute advertised network speed for effective throughput.

Short prefills, slow links and poor reuse can make separation worse. Long inputs and resource contention are reasons to investigate, not guarantees. Current SGLang documentation describes transfer paths including Mooncake and NIXL; confirm version-specific requirements.

## Check your understanding

**Question:** Do semantically identical prompts with different punctuation necessarily share the prefix cache?

<details><summary>Answer</summary>No. Reuse generally depends on exact token prefixes and execution context. Semantic similarity is not a correctness criterion for KV reuse.</details>

## Primary sources and further reading

- [SGLang paper](https://arxiv.org/abs/2312.07104)
- [SGLang PD disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)
- [SGLang request tutorial](https://docs.sglang.io/docs/basic_usage/send_request)

Sources checked on 2026-09-10. Teaching examples are not production benchmarks; confirm APIs and model support against the linked version.
