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
