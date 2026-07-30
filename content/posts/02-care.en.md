---
title: "The Deeper the Generation, the Larger the Bias Can Grow | Generative Recommendation 2026·02: CARE"
date: "2026-07-30"
updated: "2026-07-30"
category: "Generative Recommendation"
language: "en"
tags:
  - Generative Recommendation
  - Recommender Systems
  - CARE
  - Recommendation Reasoning
pairKey: "generative-recommendation-2026-02-care"
slug: "generative-recommendation-2026-02-care-en"
excerpt: "CARE interprets autoregressive Semantic ID decoding as cascaded ranking and controls prefix bias with progressive histories and query-anchored parallel reasoning."
series: "generative-recommendation-2026"
seriesOrder: 2
draft: false
---


[DIGER](/blog/generative-recommendation-2026-01-diger-en) lets recommendation loss modify item IDs, but it does not change a basic fact of autoregressive decoding:

> Once the first token is wrong, later tokens can only work diligently inside the wrong prefix.

[CARE](https://arxiv.org/abs/2602.03692) interprets this process as **cascaded ranking**. Every generated SID token shrinks the candidate set. Early decisions have the greatest power to close roads, so popular prefixes and small errors can compound layer by layer.

## Understand the paper in 30 seconds

1. **The old problem**: token generation is not a set of independent classifications. It is a sequence of irreversible candidate filters.
2. **CARE's answer**: reveal user history progressively and add query-anchored parallel reasoning branches so all computation does not roll down one prefix.
3. **What reasoning means here**: not a natural-language chain of thought, but a changed computation graph. Evaluate it through accuracy, diversity, bias, and cost.

![CARE treats autoregressive decoding as cascaded ranking](/blog/generative-recommendation-2026/02-care/mechanism.svg)

## A product tree makes prefix lock-in visible

Suppose the first token represents a broad category:

```text
sports
├── rackets
├── shoes
├── wearables
└── accessories
```

Later tokens distinguish tennis, badminton, climbing, and swimming before identifying a concrete item.

A user recently bought a swim cap but has many old racket clicks. If the first token follows the popular history into `rackets`, later layers cannot reach a quick-dry towel. The item is outside the reachable subtree.

Traditional ranking has the same pathology: an item missed by retrieval cannot be rescued by the ranker. In Semantic ID generation, each token position acts like another miniature retrieval stage.

## CARE's two structural ideas

### Progressive history encoding

CARE does not expose the full history identically at every stage. Short-term behavior can answer the current intent first, while medium- and long-term evidence enters later.

A stage representation can be written as:

\[
h_s=f_\theta(H_u^{(s)},q),
\]

where \(H_u^{(s)}\) is the history visible at stage \(s\), and \(q\) is the current request. The model does not need to resolve all evidence in its earliest decision, reducing the chance that a large long-term signal crushes a recent shift.

### Query-anchored parallel reasoning

A standard decoder predicts:

\[
p(c_j\mid H_u,c_{<j}).
\]

CARE adds branches that are anchored directly to the query and can revisit the history rather than inheriting only the previous token's conclusion. Their hidden states are fused before the final SID decision.

This is not merely a larger beam. Beam search still competes within the same prefix probability tree. Parallel reasoning changes the evidence path itself.

## Why accuracy and diversity can conflict

Popular codes have more observations, so early token choices naturally drift toward them. This may improve short-term hit rate while reducing the reachable long tail.

CARE therefore examines Recall/NDCG together with diversity- and bias-oriented measures such as DivR and ORR. The useful question is not which model has the largest single score. It is:

- Did accuracy improve by concentrating even more exposure?
- Did diversity improve only by sacrificing relevance?
- Are catalog trees and popularity distributions comparable?
- Is the added computation small enough for serving?

The paper reports relatively modest inference overhead, suggesting that structured hidden-state reasoning may be more deployable than generating a long natural-language chain of thought.

## What the paper does not prove

Popularity bias is not the whole of fairness. Supplier exposure, sensitive attributes, price discrimination, and platform incentives require separate analysis.

A hidden reasoning branch is also not an explanation. Unless a counterfactual intervention removes evidence and changes the recommendation as expected, we cannot claim the branch is a faithful causal account.

Most importantly, CARE still accepts the discrete SID tree. It helps the model walk through that tree more carefully, but it does not ask whether the tree is necessary.

The next paper, [ContRec](/blog/generative-recommendation-2026-03-contrec-en), removes the tree and generates a continuous preference vector instead of a discrete address.

[Previous: DIGER](/blog/generative-recommendation-2026-01-diger-en) · [Back to the Season 2 roadmap](/series/generative-recommendation-2026/en)
