---
title: "Scaling HSTU: ragged execution, candidate amortization and content features"
date: "2026-09-10"
updated: "2026-09-10"
category: "HSTU"
language: "en"
tags: ["hstu", "Hands-on", "Model Learning"]
pairKey: "hstu-scale"
slug: "hstu-scale-en"
excerpt: "Separate variable-length waste, candidate-dependent work and content embeddings to connect recommendation with multimodal understanding."
series: "hstu"
seriesOrder: 4
draft: false
---

![Scaling HSTU: ragged execution, candidate amortization and content features](/learning/hstu-scale-en.svg)

## Variable length is a central systems issue

For lengths `[10,100,1000]`, padding to 1000 allocates 3000 token positions but only 1110 are valid. Potential dense-attention pair work is $3\times1000^2$, compared with $10^2+100^2+1000^2$ for per-sequence work. This ratio is not a promised kernel speedup.

Ragged or jagged execution represents sequences through concatenated values and offsets, reducing padding work. Kernels must handle boundaries, grouping and balance. Stochastic Length samples sequence lengths during training; it changes the training computation rather than merely changing storage layout.

## Many candidates make reuse valuable

Target-aware ranking predicts for multiple candidates. Re-encoding history independently for every candidate duplicates work. M-FALCON involves candidate microbatching and reuse of candidate-independent computation while preserving candidate-specific dependencies. Shared caching is valid only when masks and dependencies permit it; candidate-influenced states are not automatically reusable user representations.

```python
lengths = [10,100,1000]
print('padded token positions:', len(lengths)*max(lengths))
print('ragged token positions:', sum(lengths))
print('pair-work ratio:', len(lengths)*max(lengths)**2/sum(n*n for n in lengths))
```

This computes theoretical work, not GPU timing. Profile sparse kernels, launches, caching and traffic to establish practical gains.

## Where multimodal content enters

A new item may lack interactions but already have text, images or video. A content encoder can produce features projected into the recommendation space. CLIP similarity is not user preference: visually similar items can have different behavioral value. Compare ID-only, content-only and fused models on new items, the long tail and language slices.

Content features also have temporal versions. An encoder updated using future data cannot be inserted into historical evaluation as if it existed then. Serving must account for offline/online encoding, feature freshness and missing-feature fallbacks.

## Check your understanding

**Question:** Multimodal features improve new-item metrics. Is that sufficient for rollout?

<details><summary>Answer</summary>Also inspect overall quality, established-item regressions, feature latency/missingness, cost and online outcomes. One improved slice is evidence, not the entire decision.</details>

## Primary sources and further reading

- [HSTU original paper](https://arxiv.org/abs/2402.17152)
- [Official generative-recommenders implementation](https://github.com/meta-recsys/generative-recommenders)
- [Official HSTU benchmark example](https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/README.md)

Sources checked on 2026-09-10. Teaching examples are not production benchmarks; confirm APIs and model support against the linked version.
