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
