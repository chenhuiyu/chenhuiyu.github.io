---
title: "Compute attention: Q, K, V and the causal mask"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Foundations"
language: "en"
tags: ["llm-foundations", "Hands-on", "Model Learning"]
pairKey: "llm-foundations-attention"
slug: "llm-foundations-attention-en"
excerpt: "Use four tokens to understand scores, normalization and aggregation, and why masking precedes softmax."
series: "llm-foundations"
seriesOrder: 2
draft: false
---

![Compute attention: Q, K, V and the causal mask](/learning/llm-foundations-attention-en.svg)

## A weighted lookup

A query describes what a position seeks, a key describes how a position can be matched, and a value contains the information to aggregate. This is a role analogy, not a claim that individual dimensions have human-readable meanings. Different learned projections usually produce all three from the input.

With $Q,K\in\mathbb R^{T\times d_k}$ and $V\in\mathbb R^{T\times d_v}$, first compute $S=QK^\top/\sqrt{d_k}$ with shape `[T,T]`. Normalize each row into A, then compute $O=AV$ with shape `[T,d_v]`. Weight $A_{ij}$ describes how position i reads value j. It is not a causal explanation of an answer.

## Why the square root?

If q and k dimensions are approximately independent, zero-mean and unit-variance, their dot-product variance grows with $d_k$. Dividing by its square root stabilizes the score scale and reduces early softmax saturation. This is a numerical motivation, not a theorem about every learned distribution.

```python
scores = q @ k.transpose(-2, -1) / q.shape[-1]**0.5
future = torch.ones(T, T, dtype=torch.bool).triu(1)
scores = scores.masked_fill(future, float('-inf'))
weights = scores.softmax(dim=-1)
out = weights @ v
```

This fragment assumes PyTorch tensors q/k/v and sequence length T; the notebook supplies a runnable version. Zeroing future weights after softmax without renormalizing produces rows summing below one and changes the operation.

## Mask and temperature experiment

The experiment fixes four two-dimensional Q/K vectors. Predict the first row under causal masking, then disable the mask. With one legal key, the first row is always `[1,0,0,0]`, regardless of temperature. Increasing temperature makes weights more uniform over legal positions; it must not leak into future positions.

We display a full matrix for teaching. A production implementation should avoid storing all $T^2$ entries per layer and head when possible. FlashAttention reorganizes memory access using tiles while targeting exact attention; it does not simply discard low weights.

## Check your understanding

**Question:** Does adding 100 to every legal score in each row change the result?

<details><summary>Answer</summary>No, mathematically. Softmax is invariant to a uniform row-wise shift. Implementations usually subtract the maximum to avoid exponential overflow. Different biases for different columns do change the distribution.</details>

## Primary sources and further reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [FlashAttention](https://arxiv.org/abs/2205.14135)

Sources checked on 2026-09-10. Teaching examples are not production benchmarks; confirm APIs and model support against the linked version.
