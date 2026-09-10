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

## A complete explanation from first principles

### Retrieval supplies evidence that parameters may not contain

Pretraining stores statistical structure in weights; it does not create a continuously updated company document database. If a refund policy changed yesterday, the checkpoint does not automatically know. RAG retrieves material and places it into the current input. This changes visible evidence without necessarily changing model parameters.

A practical pipeline includes parsing, chunking, indexing, query encoding, retrieval, reranking, context assembly and generation. Each stage can lose information. If PDF parsing associates a price with the wrong product, a capable generator may fluently repeat that corrupted evidence.

### Separate finding relevant text from answering correctly

Suppose a customer asks whether an item can be returned on day twenty. Two retrieved passages say “standard goods can be returned within thirty days” and “custom goods are excluded.” Both are relevant, but the system still needs the product type. Asking a clarifying question may be more appropriate than selecting the highest-similarity passage and answering yes.

Dense retrieval captures learned semantic similarity. Lexical retrieval can preserve exact model numbers, names and rare terms. Hybrid retrieval combines signals. A reranker evaluates query-document pairs more closely, at additional computational cost. Add these components in response to observed errors rather than assuming every pipeline needs every stage.

### Chunk boundaries are a modeling decision

Short chunks can separate conditions from conclusions or detach table values from headings. Long chunks may dilute retrieval signals and consume context. Overlap preserves boundary information while increasing index redundancy and repeated retrieval. Document structure and expected questions should determine the trade-off.

For multimodal documents, extracted text, table cells and image captions may depend on one another. Preserve page numbers, section paths and coordinates so answers can be traced. Updates must remove stale or deleted content; retrieving an obsolete policy is still a failure even when vector search works exactly as implemented.

### Diagnose stages with different metrics

If a query has two annotated relevant chunks and top-five retrieves one, Recall@5 is 0.5. If the first relevant chunk ranks third, its reciprocal rank is one-third. Neither score measures final-answer correctness. A lucky answer does not prove that evidence retrieval was reliable.

An oracle-context experiment supplies human-confirmed evidence directly to the generator. If it still fails, investigate generation, prompting or answer constraints. If oracle context succeeds while the actual pipeline fails, focus on retrieval, reranking and truncation. This experiment isolates a variable; it is not an extra capability available in normal production use.

### Build a trustworthy small evaluation set

For real questions, record acceptable answers, required evidence, clarification requirements and when abstention is appropriate. Split by time or source to avoid near-duplicate leakage. Average accuracy can hide failures involving dates, amounts and access permissions, so report meaningful slices separately.

LLM judges can accelerate evaluation but may react to wording, length or answer order. Calibrate them against human examples, review disagreements and preserve judging evidence. Distinguish source faithfulness from real-world correctness: an answer can faithfully repeat an outdated document and still be wrong today.

### Practice before introducing a large database

Use the actual embedding inspector to compare sentences, then change negation or a product identifier. Observe where similarity remains high despite an important distinction. Build five documents and three questions, rank them in Python and calculate Recall@k. Retain that small set when adding a real index, so infrastructure growth does not hide the behavior of individual stages.

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
