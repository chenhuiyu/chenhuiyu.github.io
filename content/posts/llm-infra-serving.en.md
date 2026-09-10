---
title: "Why serving is slow: prefill, decode, KV cache and queues"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Infrastructure"
language: "en"
tags: ["llm-infra", "Hands-on", "Model Learning"]
pairKey: "llm-infra-serving"
slug: "llm-infra-serving-en"
excerpt: "Build a latency and memory budget before discussing continuous batching, paging and prefill/decode separation."
series: "llm-infra"
seriesOrder: 1
draft: false
---

![Why serving is slow: prefill, decode, KV cache and queues](/learning/llm-infra-serving-en.svg)

## Begin with user-visible latency

Time to first token includes queueing, preprocessing, prefill and first-output overhead. Later token latency is often described by ITL or TPOT; record the exact denominator convention. High throughput does not guarantee responsiveness, particularly as queueing rises near capacity.

Prefill performs matrix operations over many tokens and often has higher arithmetic intensity. Small-batch decode generates few tokens per step and is frequently limited by weight and KV reads. This is a tendency, not a universal classification: batch size, context length, parallelism and hardware change the bottleneck.

## Budget with KV heads

For conventional MHA/GQA/MQA, approximate cache bytes are $2BTLH_{kv}D_hs$. The factor two is K plus V, and s is bytes per element. With 32 layers, 8 KV heads, dimension 128, 4096 tokens, four concurrent sequences and FP16, KV occupies 2 GiB. Exclude weights, workspace, activations and page slack from that number. Architectures such as MLA need a different model.

Use the calculator to change concurrency, length and head count. It estimates storage, not device performance. Quantized caches may require extra scales and specific kernels. Do not allocate every byte of device memory to the estimate.

## Three different optimizations

Continuous batching admits and retires requests at iteration boundaries. Paging maps logical token blocks to physical cache blocks, reducing reservation and fragmentation. Prefill/decode separation isolates execution stages into resource pools but introduces KV transfers, routing and capacity-balancing costs. These techniques can coexist; none guarantees acceleration for every workload.

## Experiment and check

Run the notebook's actual matmul benchmark after warm-up, then compare serving concurrency with fixed prompt/output lengths. Record p50/p95 TTFT, ITL, completed throughput, error rate and peak memory. Failed requests must remain visible in the denominator.

**Question:** Tokens/s increases but p95 TTFT doubles. Is the optimization successful?

<details><summary>Answer</summary>That depends on the service objective. Device utilization may improve while queueing worsens. Compare goodput satisfying latency and error constraints, not raw throughput alone.</details>

## Primary sources and further reading

- [vLLM architecture](https://docs.vllm.ai/en/latest/design/arch_overview/)
- [SGLang PD disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)

Sources checked on 2026-09-10. Teaching examples are not production benchmarks; confirm APIs and model support against the linked version.
