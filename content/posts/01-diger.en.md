---
title: "The Tokenizer Can Finally Hear Recommendation Loss | Generative Recommendation 2026·01: DIGER"
date: "2026-07-30"
updated: "2026-07-30"
category: "Generative Recommendation"
language: "en"
tags:
  - Generative Recommendation
  - Recommender Systems
  - DIGER
  - Semantic ID
pairKey: "generative-recommendation-2026-01-diger"
slug: "generative-recommendation-2026-01-diger-en"
excerpt: "DIGER keeps Semantic IDs differentiable during recommendation training and uses Gumbel exploration plus uncertainty decay to prevent codebook collapse."
series: "generative-recommendation-2026"
seriesOrder: 1
draft: false
---


Season 1 ended with [OneReason](/blog/generative-recommendation-20-onereason-en): a model could generate item codes, recommendation rationales, and even an explicit reasoning trace. Yet one foundation was still mostly untouched by the downstream task:

> The Semantic ID of an item is usually frozen before the recommender begins training.

It is like asking a brick factory to optimize only for texture reconstruction, while the architect must accept whatever bricks arrive. Beautiful bricks are not necessarily useful for a building. [DIGER](https://arxiv.org/abs/2601.19711) asks a direct question: **can recommendation loss change the item language itself?**

## Understand the paper in 30 seconds

1. **The old mismatch**: an RQ-VAE learns Semantic IDs through content reconstruction, while the recommender learns next-item behavior. The recommendation gradient stops before the tokenizer.
2. **DIGER's answer**: keep the forward pass discrete, but backpropagate through Gumbel-Softmax probabilities so downstream loss can update the codebook.
3. **The real difficulty**: differentiability is not stability. A naive straight-through estimator can commit too early to a few codes and collapse the codebook. DIGER uses early exploration and later uncertainty decay.

![DIGER lets recommendation gradients cross the Semantic ID boundary](/blog/generative-recommendation-2026/01-diger/mechanism.svg)

## See the mismatch with eight toy items

Reuse the season's toy catalog: tennis racket, tennis ball, climbing shoes, chalk, badminton racket, shuttlecock, swim cap, and quick-dry towel.

By content, the tennis and badminton rackets look similar. By behavior, climbing shoes and chalk may be purchased consecutively. A conventional two-stage pipeline first asks:

```text
Which items look most alike in text and images?
```

The recommender must then adapt to those fixed IDs. DIGER lets the tokenizer hear a second question:

```text
Which encoding makes the next user choice easier to predict?
```

Content similarity remains a constraint, but it is no longer the only authority over the item language.

## What happens from input to output?

For item \(v\), code position \(j\), and candidate code \(i\), the encoder computes:

\[
\ell_{v,j,i}=\operatorname{sim}(r_{v,j},e_i).
\]

Gumbel noise produces a soft categorical distribution:

\[
\tilde y_{v,j,i}
=
\frac{\exp((\ell_{v,j,i}+g_{v,j,i})/\tau)}
{\sum_{k=1}^{K}\exp((\ell_{v,j,k}+g_{v,j,k})/\tau)}.
\]

The forward pass still chooses a hard code:

\[
c_{v,j}=\arg\max_i(\ell_{v,j,i}+g_{v,j,i}),
\]

so the recommender receives real discrete tokens. During backpropagation, however, DIGER uses a soft embedding:

\[
\bar e_{v,j}=\sum_i\tilde y_{v,j,i}e_i.
\]

Every candidate code can receive a probability-weighted gradient instead of updating only the argmax winner. The bridge has two decks: discrete forward computation and differentiable backward learning.

## Why the noise matters

With deterministic hard assignments at the start of training, a few codes may win by chance. Winning codes receive more updates, become even more attractive, and turn the codebook into a crowded city with many abandoned addresses.

DIGER's DRIL module explores alternative assignments early, then reduces randomness later. The paper proposes two schedules:

- **SDUD** links the noise scale to the falling generation loss, so assignments become more deterministic as training stabilizes.
- **FrqUD** adds exploration around overused codes while leaving low-frequency codes closer to deterministic inference.

The point is not maximum noise. Early training needs search; late training needs closure.

## Training versus inference

Training jointly updates the tokenizer and the generator with recommendation, reconstruction, and quantization-related objectives. Inference removes Gumbel sampling and uses stable hard SIDs.

This opens several engineering questions:

- How do historical logs remain compatible after a codebook update?
- Can new items be encoded locally without rebuilding the catalog?
- How do caches, indexes, and model versions migrate together?
- Does a better offline codebook preserve its gain under large beam search?

## What do the experiments prove?

The paper evaluates B-Shop, I-Shop, Yelp, and related public setups using Recall@10, NDCG@10, code utilization, and training stability. The evidence supports two claims:

1. Letting recommendation gradients influence SIDs can outperform a fully frozen tokenizer.
2. Differentiability alone is insufficient; exploration and code balance determine whether training survives.

The experiments do not establish the cost of frequently changing IDs in a live catalog, nor do they prove that end-to-end re-indexing remains stable under production traffic.

## The contribution that matters most

DIGER changes the module boundary. The tokenizer is no longer an offline preprocessing tool. It becomes part of the recommender.

That shift also makes the system more fragile. Once recommendation loss can rewrite item IDs, indexing, logging, caching, and experiment comparability enter the training loop. The research question becomes:

> How can an item language keep learning without forcing the entire platform to speak a new dialect every day?

The next paper, [CARE](/blog/generative-recommendation-2026-02-care-en), takes another unresolved problem: even with better IDs, token-by-token generation can amplify one early mistake into an entire wrong path.

[Back to the Season 2 roadmap](/series/generative-recommendation-2026/en)
