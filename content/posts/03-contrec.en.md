---
title: "Semantic IDs Are Not the Only Answer | Generative Recommendation 2026·03: ContRec"
date: "2026-07-30"
updated: "2026-07-30"
category: "Generative Recommendation"
language: "en"
tags:
  - Generative Recommendation
  - Recommender Systems
  - ContRec
  - Diffusion
pairKey: "generative-recommendation-2026-03-contrec"
slug: "generative-recommendation-2026-03-contrec-en"
excerpt: "ContRec abandons discrete Semantic IDs, generates a continuous preference vector with diffusion, and retrieves a real catalog item afterward."
series: "generative-recommendation-2026"
seriesOrder: 3
draft: false
---


[CARE](/blog/generative-recommendation-2026-02-care-en) builds better bridges inside a discrete SID tree. [ContRec](https://arxiv.org/abs/2504.12007) asks a less polite question:

> Why must generative recommendation quantize items into discrete tokens at all?

Semantic IDs let Transformers generate products like words, but quantization removes detail. Codebook size, collisions, sequence length, and decoding cost remain entangled. ContRec chooses another route: **generate a preference in continuous space, then map it back to the catalog.**

## Understand the paper in 30 seconds

1. **The old problem**: discrete quantization cuts a continuous item manifold into cells. Longer SIDs improve expressiveness but increase decoding cost.
2. **ContRec's answer**: keep a non-quantized item representation and use conditional diffusion to generate the next-item preference vector.
3. **The cost moves elsewhere**: no codebook collapse, but now the system pays for iterative denoising, vector retrieval, and real-time index maintenance.

![ContRec moves from discrete addresses to continuous preferences](/blog/generative-recommendation-2026/03-contrec/mechanism.svg)

## What is a continuous token?

A discrete representation might encode:

```text
climbing shoes  <sport><shoe><climb>
chalk           <sport><gear><climb>
```

The hierarchy is readable, but color, stiffness, price, and use context are compressed into a finite code. ContRec keeps a continuous embedding \(e_i\in\mathbb R^d\) instead.

The user history becomes a condition \(c_u\). The model no longer predicts a few codes. It generates a vector \(\hat e\) that should land near the next item the user may prefer.

## How diffusion generates a recommendation

Training adds noise to the target item embedding \(e_0\):

\[
e_t=\sqrt{\bar\alpha_t}e_0+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
\]

The denoiser receives the noisy item, timestep, and user condition:

\[
\hat\epsilon_\theta=\epsilon_\theta(e_t,t,c_u),
\]

and minimizes:

\[
\mathcal L_{\text{diff}}
=
\mathbb E\left[\lVert\epsilon-\hat\epsilon_\theta\rVert_2^2\right].
\]

Inference starts from Gaussian noise, repeatedly denoises to obtain \(\hat e_0\), and retrieves a real item:

\[
\hat i=\arg\max_{i\in\mathcal I}\operatorname{sim}(\hat e_0,e_i).
\]

ContRec therefore does not invent a product. It generates a continuous query whose final answer is constrained by the catalog.

## Where the LLM enters

The framework also uses an LLM-oriented preference representation as a conditioning signal. The attractive division of labor is clear: language summarizes the history, while diffusion finds a point in item space.

But a natural-language preference statement is not automatically a faithful explanation. To show that “the user is moving toward climbing” truly matters, the system should remove or alter that condition and verify that the generated vector moves accordingly.

## How to read the experiments

ContRec evaluates public datasets including LastFM, MovieLens-1M, Amazon Beauty, and Games against sequential, diffusion, LLM, and SID baselines.

Three dimensions must remain separate:

- **Ranking quality**: HR and NDCG.
- **Generation cost**: denoising steps, latency, and GPU throughput.
- **Retrieval cost**: nearest-neighbor search over a changing catalog.

Fewer diffusion steps may reduce quality. More steps may erase the advantage over a single autoregressive pass. A production comparison must include both generation and vector-index latency.

## Where it fails

Continuous space avoids invalid SID sequences, but it can generate a vector between products. Nearest-neighbor mapping may return an item that is semantically close yet unavailable, overpriced, duplicated, or operationally ineligible.

The method also naturally targets one next-item vector. Generating a page requires either multiple independent samples, which may conflict or repeat, or a joint high-dimensional sample, which is more expensive.

ContRec's most important contribution is not “diffusion beats Transformers.” It demonstrates that:

> Semantic ID is an engineering interface, not a logical law of generative recommendation.

The next paper, [DualFashion](/blog/generative-recommendation-2026-04-dualfashion-en), pushes continuous generation further. The model no longer generates a query for an existing product; it generates a product image and description that may not exist yet.

[Previous: CARE](/blog/generative-recommendation-2026-02-care-en) · [Back to the Season 2 roadmap](/series/generative-recommendation-2026/en)
