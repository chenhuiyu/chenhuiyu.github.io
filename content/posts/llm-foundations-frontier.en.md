---
title: "Toward the frontier: MoE, long context and verified reasoning"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Foundations"
language: "en"
tags: ["llm-foundations", "Hands-on", "Model Learning"]
pairKey: "llm-foundations-frontier"
slug: "llm-foundations-frontier-en"
excerpt: "Understand frontier designs through compute, memory, communication and verification rather than model names alone."
series: "llm-foundations"
seriesOrder: 8
draft: false
---

![Toward the frontier: MoE, long context and verified reasoning](/learning/llm-foundations-frontier-en.svg)

## The frontier reallocates bottlenecks

“SOTA” is not a permanent winner. Tasks, budgets and software versions change. A more durable reading method asks what decreases, where the cost moves and what experiment could refute the claimed benefit. Record training compute, activated parameters, total parameters, context length and measured throughput separately.

## MoE: fewer expert computations, more routing concerns

A mixture-of-experts router typically selects a subset of FFN experts per token. Total parameters govern storage requirements; activated parameters are closer to per-token compute, but are not identical to FLOPs or latency. Load imbalance, cross-device all-to-all and small matrix inefficiency can erase savings. Expert parallelism is different from tensor parallelism.

Selecting two of eight experts activates one quarter of the expert FFNs, not one quarter of the entire model. Attention, routing, shared experts and communication remain. Report load distributions and token-drop or capacity policies.

## Long context: capacity is not effective use

GQA reduces KV head count; MLA uses architectural mechanisms including latent compression. The same cache formula cannot be applied blindly to both. Position extension, retrieval, compression and sparse attention address different limitations. Test evidence position, distractors and reasoning depth while recording time to first token and cache cost.

## More reasoning requires verification

Repeated sampling, search, explicit reasoning and tool use allocate more compute at test time. Code or mathematics may offer verifiable feedback, but a verifier only covers its specification: passing tests does not rule out untested bugs. Report generated-token budgets and pass@k separately. Comparing k=64 against k=1 does not isolate model quality.

```python
# Independent, identical trials: an illustrative probability, not a real estimator.
p = 0.2
for k in [1, 4, 16]:
    print(k, 1 - (1-p)**k)
```

Real attempts can be correlated, and finite-sample pass@k evaluation needs the appropriate estimator. This formula only illustrates why additional attempts alone raise the chance of at least one success.

## Check your understanding

**Question:** Total parameters double while activated parameters stay fixed. Does inference memory stay fixed?

<details><summary>Answer</summary>Not necessarily. Additional expert weights still require storage, sharding or transfers. Activated parameters describe routed computation, not the total weight and communication budget.</details>

## Primary sources and further reading

- [DeepSeek-V2 / MLA and MoE](https://arxiv.org/abs/2405.04434)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)

Sources checked on 2026-09-10. Teaching examples are not production benchmarks; confirm APIs and model support against the linked version.
