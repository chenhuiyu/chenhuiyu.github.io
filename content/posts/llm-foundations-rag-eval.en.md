---
title: "Embeddings, RAG and evaluation: retrieval is not an answer"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Foundations"
language: "en"
tags: ["llm-foundations", "Hands-on", "Model Learning"]
pairKey: "llm-foundations-rag-eval"
slug: "llm-foundations-rag-eval-en"
excerpt: "Measure retrieval, reranking, context assembly and answer attribution separately, then improve the failing stage."
series: "llm-foundations"
seriesOrder: 7
draft: false
---

![Embeddings, RAG and evaluation: retrieval is not an answer](/learning/llm-foundations-rag-eval-en.svg)

## Trace one RAG request

A question such as “What is this service's timeout?” becomes a query vector, retrieves document chunks, passes through reranking and context assembly, and finally reaches a generator that should cite supporting evidence. Failure can occur at corpus coverage, chunk boundaries, retrieval, ranking, truncation or evidence use.

Calling every failure hallucination obscures the fix. Record intermediate inputs and outputs, then find the first stage where the correct evidence disappeared. RAG supplies updateable evidence; it does not guarantee a correct answer.

## Retrieval shapes and limits

Documents form `[N,D]`, queries `[B,D]`, and dot products `[B,N]`. With normalized vectors, dot product equals cosine. The index distance must match representation training and normalization. Approximate nearest-neighbor search exchanges exactness for latency and memory; index parameters affect Recall@k. The embedding objective may also mismatch the task.

```python
import torch
import torch.nn.functional as F
docs = F.normalize(torch.tensor([[1.,0.],[0.,1.],[1.,1.]]), dim=-1)
query = F.normalize(torch.tensor([[.9,.1]]), dim=-1)
scores = query @ docs.T
print(scores.topk(2, dim=-1).indices)
```

This is a geometry demonstration, not a semantic model. Use the preceding chapter for real embeddings.

## Give each stage a falsifiable metric

| Stage | Metric | Diagnostic question |
|---|---|---|
| Corpus | Evidence coverage | Does the snapshot contain the answer? |
| Retrieval | Recall@k | Does the relevant chunk enter candidates? |
| Reranking | MRR / NDCG | Is useful evidence near the top? |
| Generation | Correctness, citation support | Does the source support the claim? |
| Serving | p95 latency, failure rate | Does the user reliably receive a result? |

Fix corpus snapshot, question set, token budget and latency constraints when comparing systems. Do not attribute extra-context gains to a new retriever. Calibrate model-based judges against human samples and examine position bias, verbosity preference and correlated errors.

## Check your understanding

**Question:** Can Recall@20 rise while final correctness falls?

<details><summary>Answer</summary>Yes. Noise can consume the context budget, introduce conflicts or distract the generator. Reranking or prompt changes may also be responsible. Stage-level traces locate the failure.</details>

## Primary sources and further reading

- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [BEIR](https://arxiv.org/abs/2104.08663)

Sources checked on 2026-09-10. Teaching examples are not production benchmarks; confirm APIs and model support against the linked version.
