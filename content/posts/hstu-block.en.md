---
title: "HSTU tensor by tensor: SiLU aggregation, temporal bias and gating"
date: "2026-09-10"
updated: "2026-09-10"
category: "HSTU"
language: "en"
tags: ["hstu", "Hands-on", "Model Learning"]
pairKey: "hstu-block"
slug: "hstu-block-en"
excerpt: "Compare softmax attention with pointwise aggregation and trace U/Q/K/V, normalization and residual paths."
series: "hstu"
seriesOrder: 2
draft: false
---

![HSTU tensor by tensor: SiLU aggregation, temporal bias and gating](/learning/hstu-block-en.svg)

## A complete explanation from first principles

### Begin with the block's function

HSTU means Hierarchical Sequential Transduction Unit. Treat it first as a function that transforms behavior-sequence representations, then inspect projections, aggregation, normalization and gating. Replacing Transformer softmax with another function alone does not reproduce the full design.

Efficient implementations may use ragged storage rather than dense `[B,T,d]`. This chapter uses a fixed-length sequence for inspection. Logical tensor relationships and physical storage are separate layers of understanding.

### Follow the U, V, Q and K branches

Learned projections produce branches with different roles: Q/K form position-matching scores, V supplies aggregated content and U participates in gating. An implementation may obtain them through one larger projection followed by slicing, while retaining distinct parameter subspaces.

A schematic flow forms biased QK scores, applies a SiLU-like transformation and valid-position constraints, aggregates V, normalizes, multiplies by a U-related branch, projects and adds a residual. Exact scales, normalization and layouts belong to the paper and implementation; the browser example explicitly omits components.

### SiLU aggregation coefficients are not probabilities

SiLU is `x*sigmoid(x)`. At -1 it is approximately -0.269, at zero it is zero, and at one it is approximately 0.731. Unlike softmax, it does not produce nonnegative row-normalized coefficients.

With values `[1,0]` and `[0,1]`, those coefficients yield `[-0.269,0.731]`, outside their convex-combination line segment. Negative coefficients can subtract feature directions, but do not directly mean dislike for an item. Gating and output projections further change the representation.

### Do not copy softmax masking mechanically

Softmax commonly masks illegal scores with negative infinity before normalization. Directly applying SiLU to negative infinity can involve infinity multiplied by zero and produce NaN. A valid implementation can zero illegal aggregation coefficients after the nonlinearity or use another verified treatment.

Distinguish history, padding, prediction and candidate positions. A triangular-looking mask is not a full correctness test. Change future inputs and verify that earlier outputs remain unchanged.

### Relative time adds information content similarity lacks

Two interactions with the same item can have different relevance when one happened a minute ago and the other a month ago. Relative-time or position biases make such distinctions expressible without declaring all old behavior useless.

Units, bucketing and truncation matter. Training in seconds but serving milliseconds can place events in different bias regions. Input-contract errors may be less visible than formula errors.

The experiments check signed coefficients, the effect of changing temporal bias on actual aggregates, and causal/padding boundaries. The browser notebook fixes embeddings and parts of the aggregator, then trains a scoring head. The full PyTorch notebook offers a more complete but still educational HSTU-inspired block with end-to-end training. Neither claims to reproduce an entire production recommendation system. Explicit omissions make each experiment's evidence easier to interpret.

## Similar shape, different aggregation

Standard attention softmax-normalizes scores across each row. HSTU includes pointwise activated aggregation: transform biased scores with a function such as SiLU, aggregate V, then normalize and gate. Without sequence-wise softmax, the weight matrix is not a probability distribution.

Our single-head teaching equation is $A_{ij}=\mathbf1_{j\le i}\,\mathrm{SiLU}(q_i^\top k_j+b_{ij})/N$, followed by $o_i=W_o[U_i\odot\mathrm{Norm}(\sum_jA_{ij}V_j)]+x_i$. The demo uses fixed sequence length N. Check official code for actual scaling, head layout, bias and normalization.

## Trace four projections

Normalized `[B,T,D]` inputs produce U/V/Q/K projections. Q/K match positions, V supplies aggregated values and U gates the output. Q/K head dimensions can differ from U/V dimensions. Fusing projections can reduce operator overhead, but the split must match weight layout.

```python
import torch
import torch.nn.functional as F
q, k, v = [torch.randn(1,4,8) for _ in range(3)]
mask = torch.ones(4,4,dtype=torch.bool).tril()
a = F.silu(q @ k.transpose(-2,-1)) / 4
a = a.masked_fill(~mask, 0)
y = a @ v
print(a.sum(-1))  # Not required to equal one.
```

SiLU can produce negative coefficients. Probability entropy cannot be applied directly to this matrix without defining a different meaningful measure.

## Temporal bias is not simply fixed forgetting

Relative positions and elapsed time distinguish three consecutive events from three events spread over months. Real bias parameterization includes learned implementation-specific details. The interactive $-\alpha(i-j)$ bias is deliberately simple and is not the paper's complete construction.

Increasing α can make old-position weights negative. Combined with normalization and gating, effects on outputs need not be monotonic. Inspect outputs and ablations rather than guessing recommendation behavior from heatmap colors.

## Check your understanding

**Question:** Does replacing softmax with SiLU fully implement HSTU?

<details><summary>Answer</summary>No. Data formulation, projections, gating, normalization, temporal/position bias and system optimizations also matter. The notebook is explicitly HSTU-inspired teaching code, not a complete reproduction.</details>

## Primary sources and further reading

- [HSTU original paper](https://arxiv.org/abs/2402.17152)
- [Official generative-recommenders implementation](https://github.com/meta-recsys/generative-recommenders)

Sources checked on 2026-09-10. Teaching examples are not production benchmarks; confirm APIs and model support against the linked version.
