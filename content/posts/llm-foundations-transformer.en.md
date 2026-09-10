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

## A complete explanation from first principles

### Establish the block's shape contract

A Transformer block commonly receives `[B,T,d]` and returns `[B,T,d]`. Batch size, sequence length and model width stay constant; the representation at each position changes. This shape contract allows blocks to be stacked. More layers mean repeated processing of the same positions, not a longer sentence.

One common pre-norm layout is `h = x + Attention(Norm(x))`, followed by `y = h + FFN(Norm(h))`. Other layouts exist, including post-norm and parallel residual arrangements. Learn one concrete layout, then inspect the actual model configuration instead of treating a diagram as universal.

### A residual learns a correction

For input `[1,2]` and a module correction `[0.1,-0.2]`, residual addition gives `[1.1,1.8]`. If the module initially contributes little, the original signal still has a direct route through the block. A layer does not need to reconstruct the entire representation from scratch.

For `y=x+f(x)`, the local Jacobian is `I+J_f`. The identity term creates a direct gradient path, but does not prove that arbitrarily deep networks are easy to optimize. Initialization, learning rate, normalization and numerical precision still matter.

### Normalization operates over a particular axis

LayerNorm usually computes statistics across the features of each individual position. For `[1,2,3]`, the mean is two; centering gives `[-1,0,1]`, which is divided by a stabilized standard deviation and then transformed by learned scale and offset. RMSNorm instead scales by a root-mean-square quantity and typically does not subtract the mean.

Neither operation substitutes for attention: a per-position normalization does not read other token positions. Its role is to control feature scale. The small epsilon in the denominator prevents numerical problems and should not be casually removed.

### The feed-forward network contributes substantial capacity

Attention mixes information across positions. The FFN nonlinearly transforms the information already collected at each position. A simple FFN expands the feature dimension, applies an activation and projects back down. With width 512 and intermediate width 2,048, the two matrices together contain approximately 2.1 million parameters, excluding biases. Focusing only on attention misses much of a block's parameter and compute budget.

Without a nonlinearity, the two linear maps could collapse into one matrix. A gated FFN adds a branch that modulates another branch's features. SwiGLU uses a SiLU-based gate, but this is different from routing tokens to separate experts in a mixture-of-experts layer.

### Position has to enter the computation

Content similarity alone does not adequately express the difference between “dog chases cat” and “cat chases dog.” Absolute position embeddings add position-dependent vectors. RoPE rotates paired query/key coordinates so their dot product incorporates relative position relationships. It is not simply a timestamp appended to the input text.

A longer supported input window also raises questions beyond producing position numbers: which distances appeared in training, how rotation frequencies are handled, and whether evidence in the middle of the input remains usable. Accepting a longer tensor is weaker evidence than demonstrating reliable long-range reasoning.

### Inspect one layer before scaling the network

In the model I/O notebook, compare embedding outputs, intermediate hidden states and final logits. Check that residual operands match in shape, normalization uses the intended feature axis, and the FFN returns to the original width. Compare a repeated token across different contexts to distinguish its fixed lookup embedding from its contextual representation.

If training produces NaNs, locate the first non-finite activation or gradient rather than blindly reducing the learning rate. If more layers fail to improve quality, investigate data, supervision and optimization as well as architecture. Parameter count alone does not identify the limiting factor.

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
