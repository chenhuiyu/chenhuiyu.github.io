---
title: "Train a tiny HSTU-inspired recommender: data, loss and top-k"
date: "2026-09-10"
updated: "2026-09-10"
category: "HSTU"
language: "en"
tags: ["hstu", "Hands-on", "Model Learning"]
pairKey: "hstu-training"
slug: "hstu-training-en"
excerpt: "Train executable PyTorch teaching code and verify causality, padding, gradients and recommendation outputs."
series: "hstu"
seriesOrder: 3
draft: false
---

![Train a tiny HSTU-inspired recommender: data, loss and top-k](/learning/hstu-training-en.svg)

## Close the loop even for a tiny model

The notebook needs neither private logs nor a paid GPU. It generates controlled behavior patterns, trains a single-head model with U/Q/K/V, SiLU aggregation, gating and residuals, then produces next-item logits. Synthetic data and simplified temporal structure teach mechanics; they do not establish production impact.

IDs `[B,T]` become embeddings `[B,T,D]`, the block preserves that shape, and an output head produces `[B,T,V]`. Every valid position can supervise next-item prediction. Padding target zero is excluded with `ignore_index=0`.

## One backward pass is not learning evidence

Fix the seed and train deterministic sequences while plotting loss. Inspect gradient norms and compare top-k before and after training. Then test retained sequences with different starts and report actual metrics. Success on an easy synthetic rule only establishes learning within that controlled task.

```python
# Uses model / inputs / targets defined in the complete notebook.
logits = model(inputs)
loss = torch.nn.functional.cross_entropy(
    logits.flatten(0,1), targets.flatten(), ignore_index=0)
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

This is a training-step fragment. The notebook supplies model definitions, data, optimizer, loop, plots and assertions.

## Three essential correctness checks

Changing a future token must not alter earlier outputs: test causality. Replacing padding embeddings with nonzero random values must not affect legal outputs through masked padding. Exclude reserved ID zero before top-k evaluation.

Separate falling training loss from improving validation. A training-set-only result is an implementation sanity check, not generalization. Even the notebook's held-out synthetic patterns are not a real-world independent test set.

## Check your understanding

**Question:** Loss is low but recommendations always return ID zero. What do you inspect first?

<details><summary>Answer</summary>Check padding supervision, valid targets, candidate filtering and indexing of the final valid position. You may be reading a trailing padding position instead of the sequence's real endpoint.</details>

## Actual output from this run

![Actual training curve for the 3,538-parameter teaching model. Synthetic convergence does not establish industrial recommendation quality.](/learning/hstu-training-actual.png)

Actual training curve for the 3,538-parameter teaching model. Synthetic convergence does not establish industrial recommendation quality.

## Primary sources and further reading

- [Official generative-recommenders implementation](https://github.com/meta-recsys/generative-recommenders)
- [PyTorch cross-entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

Sources checked on 2026-09-10. Teaching examples are not production benchmarks; confirm APIs and model support against the linked version.
