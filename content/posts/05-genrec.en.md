---
title: "One Next Item Is Not Enough: Generate the Whole Page | Generative Recommendation 2026·05: GenRec"
date: "2026-07-30"
updated: "2026-07-30"
category: "Generative Recommendation"
language: "en"
tags:
  - Generative Recommendation
  - Recommender Systems
  - GenRec
  - Reinforcement Learning
pairKey: "generative-recommendation-2026-05-genrec"
slug: "generative-recommendation-2026-05-genrec-en"
excerpt: "GenRec trains and generates at page level, compresses long histories with Token Merger, and aligns page preferences through GRPO-SR."
series: "generative-recommendation-2026"
seriesOrder: 5
draft: false
---


The first four papers focused on representation, reasoning, and what may be generated. [GenRec](https://arxiv.org/abs/2604.14878) returns to the real application surface:

> When a user scrolls, the system needs a page, not a single next item.

The same history can support many reasonable pages. If every clicked item becomes the unique label for the same input, a normal multi-answer problem looks like contradictory supervision. GenRec makes the page itself the unit of learning.

## Understand the paper in 30 seconds

1. **The old problem**: next-item training ignores order, complementarity, repetition, and diversity within a list. Long histories serialized as multi-token SIDs also consume context and compute.
2. **GenRec's answer**: Page-wise Generation predicts a full page; Token Merger compresses history; GRPO-SR aligns relevance, validity, and preference through page-level rewards.
3. **The important evidence**: the paper reports large-scale deployment and a month-long online experiment. The evidence is valuable, while private data, reward models, and internal baselines limit independent verification.

![GenRec moves from next-item labels to page generation](/blog/generative-recommendation-2026/05-genrec/mechanism.svg)

## Why point-wise supervision conflicts

Suppose a sports page displays and receives engagement on:

```text
climbing shoes → chalk → quick-dry towel → swim cap
```

Four separate next-item samples would ask the same history to have four different unique answers. Page-level supervision preserves the fact that the products formed one presentation and that their order and co-occurrence carry policy information.

The objective becomes:

\[
p(S_u\mid H_u)
=
\prod_{j=1}^{|S_u|}
p(i_j\mid H_u,i_{<j}),
\]

where \(S_u\) is the complete page. Later items depend on earlier generated items, so the model can learn de-duplication, complementarity, and page rhythm.

## What Token Merger compresses

If every historical item is represented by several SID tokens, hundreds of actions become thousands of tokens. Token Merger combines local item or history tokens into a more compact representation.

The compression mainly reduces **history encoding**. The page still requires autoregressive output. When a paper says token count is halved, ask whether it halves the prompt, KV cache, or final decoding steps. These are not interchangeable savings.

## Why reinforcement learning is still needed

Supervised learning imitates logged pages. It does not know whether a newly generated page is good. GRPO-SR compares sampled pages with a page-level reward that can be simplified as:

\[
R(S)=R_{\text{valid}}
+
\mathbb 1[R_{\text{valid}}>0]
\left(
\alpha R_{\text{relevance}}
+\beta R_{\text{preference}}
+\gamma R_{\text{diversity}}
\right).
\]

The validity gate matters. A page containing illegal items, severe repetition, or broken structure should not compensate with a high relevance score. The gate blocks a common form of reward hacking.

Reward design also writes platform values into the model. Different weights on relevance, long-tail exposure, conversion, and satisfaction produce different pages.

## How to read the online result

The paper reports gains in clicks, transactions, and long-tail outcomes and describes deployment in a large application. That supports the value of the complete system, not necessarily every module in isolation.

A production change often includes:

- page labels and sample construction;
- tokenizer and multimodal features;
- model scale;
- RL reward;
- beam search, caching, and filtering;
- traffic policy and catalog eligibility.

The online A/B test proves that the package can work. Module ablations are still needed to identify what transfers elsewhere.

## Where it fails

Page generation faces three hard issues:

1. **Multiple valid answers**: logs reveal one exposed page, not all good pages.
2. **Policy bias**: clicks come from an old exposure policy and may reproduce its blind spots.
3. **Decoding constraints**: inventory, duplication, compliance, and live state must be enforced during or after generation.

GenRec shows that page-level generation can enter production. Advertising adds budgets, bids, expected revenue, and competition.

The next paper, [GR4AD](/blog/generative-recommendation-2026-06-gr4ad-en), puts the generator into the advertising mainline, where each token can change revenue directly.

[Previous: DualFashion](/blog/generative-recommendation-2026-04-dualfashion-en) · [Back to the Season 2 roadmap](/series/generative-recommendation-2026/en)
