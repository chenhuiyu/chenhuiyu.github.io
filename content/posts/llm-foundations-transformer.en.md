---
title: "Beyond attention: residuals, normalization, FFNs and position"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Foundations"
language: "en"
tags: ["llm-foundations", "Hands-on", "Model Learning"]
pairKey: "llm-foundations-transformer"
slug: "llm-foundations-transformer-en"
excerpt: "Trace a decoder block and distinguish token mixing from feature transformation."
series: "llm-foundations"
seriesOrder: 3
draft: false
---

![Beyond attention: residuals, normalization, FFNs and position](/learning/llm-foundations-transformer-en.svg)

## Two different transformations

Attention mixes information across token positions. An FFN applies the same learned transformation independently at each position, changing features. Calling a Transformer a stack of attention layers misses much of its parameter count and computation. A common pre-norm block is $h=x+\mathrm{Attn}(\mathrm{Norm}(x))$, followed by $y=h+\mathrm{FFN}(\mathrm{Norm}(h))$. Actual architectures differ in normalization, biases and residual placement.

Residual paths let a layer learn an increment and provide a more direct gradient route. They do not guarantee stability: initialization, learning rate and depth still matter. LayerNorm centers and rescales; RMSNorm primarily rescales by the root mean square. They are not identical.

## Why does the FFN expand first?

A conventional FFN projects D features into a wider M, applies a nonlinearity and projects back to D, preserving the shape required for residual addition. SwiGLU adds a gating branch: $\mathrm{FFN}(x)=W_o[\mathrm{SiLU}(W_gx)\odot W_ux]$. This equation uses column-vector notation; verify multiplication orientation in code.

An input `[B,T,D]` expands to `[B,T,M]`. A two-projection FFN has approximately $2DM$ weights; a three-projection gated FFN has approximately $3DM$. Holding M constant does not preserve parameter count.

## What supplies position?

Without position information or asymmetric masking, self-attention is permutation-equivariant: permuting inputs permutes outputs. RoPE rotates pairs of Q/K coordinates so dot products encode relative positions. It is not simple addition of a position vector and does not automatically provide unlimited context extrapolation.

```python
import torch
x = torch.randn(2, 5, 8)
norm = torch.nn.LayerNorm(8)
ffn = torch.nn.Sequential(torch.nn.Linear(8, 32),
                          torch.nn.GELU(), torch.nn.Linear(32, 8))
y = x + ffn(norm(x))  # Only the FFN residual branch.
assert y.shape == x.shape
```

## Inspect the actual structure

In the real-model notebook, print the layer count, attention heads, hidden size and hidden-state shape at every layer. Set evaluation mode before repeated inference, so training-time dropout is not mistaken for sampling randomness.

**Question:** If an FFN never mixes positions, how can its output reflect context?

<details><summary>Answer</summary>Its input already includes context mixed by attention. Alternating position mixing and within-position nonlinear transformations creates the overall computation.</details>

## Primary sources and further reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)

Sources checked on 2026-09-10. Teaching examples are not production benchmarks; confirm APIs and model support against the linked version.
