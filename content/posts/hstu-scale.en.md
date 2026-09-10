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

## A complete explanation from first principles

### Scale exposes uneven lengths and repeated work

Histories range from a few events to thousands. Padding every sequence to the longest history wastes positions. Separately, many candidates for one user share nearly the same historical computation. Ragged execution and candidate amortization address these different redundancies.

Removing padding does not automatically turn quadratic attention into linear attention. Valid positions still follow the model's connectivity and kernel design.

### Calculate offsets for a packed batch

Lengths two, five and one use fifteen positions in a padded batch but only eight valid positions. Ragged storage concatenates values and records offsets `[0,2,7,8]`. The third sequence is `values[7:8]`.

```python
sequences = [[1,2],[3,4,5,6,7],[8]]
values, offsets = [], [0]
for sequence in sequences:
    values.extend(sequence)
    offsets.append(len(values))
print(values, offsets)
assert all(values[offsets[i]:offsets[i+1]] == s for i,s in enumerate(sequences))
```

This validates storage only. Attention must preserve user boundaries rather than treating concatenated values as one history. Empty sequences, very long histories and offset datatypes need additional handling.

### Shared computation depends on candidate dependence

If history representation h is candidate-independent, compute it once and score many candidates in a batch. If candidates interact with history early in the network, the reusable region is smaller. Analyze the actual candidate arrangement and attention masks rather than assuming every ranker can cache one universal user vector.

Separate history encoding, candidate-conditioned work, final scoring and communication. Even perfect reuse of an eighty-percent historical component leaves candidate-dependent work that grows with candidate count. Memory access and batching also affect realized gains.

### Content features provide a cold-start information path

New items may lack interactions while already having titles, images or videos. A content encoder supplies semantic features that can be projected, concatenated or gated together with ID embeddings. That gives the model evidence about new items, but similar content does not guarantee similar click or purchase behavior.

Price, availability, popularity and user intent also matter. A CLIP vector is not automatically calibrated for a recommendation objective. Compare ID-only, content-only and fused variants, separating new-item and established-item results.

### Feature versions form a production interface

Encoder revision, preprocessing and normalization must remain compatible between training and serving. Changing encoders for only part of a catalog can mix incompatible spaces even when vector dimensions match. Record revisions, generation times and schemas, and plan recomputation and transitions.

Training features should also reflect what was available at the historical decision time. Content understanding, sequence modeling and retrieval systems jointly own this interface; it is more than an anonymous float array.

Evaluate efficiency through valid-token throughput, memory, padding, candidates and latency; quality through temporal splits and cold-start ranking; reliability through empty histories, removed items, missing features and version transitions. The browser exercises validate small-scale offsets and scoring logic. Actual ragged kernels and distributed throughput require hardware measurements. Establish user boundaries and candidate dependencies before claiming an acceleration.

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
