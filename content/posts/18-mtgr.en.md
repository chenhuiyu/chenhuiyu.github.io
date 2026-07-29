---
title: "Should We Really Throw Away Every Traditional Cross Feature? | Generative Recommendation 18: MTGR"
date: "2026-07-29"
updated: "2026-07-29"
category: "Generative Recommendation"
language: "en"
tags:
  - Generative Recommendation
  - Recommender Systems
  - MTGR
  - Industrial Ranking
pairKey: "generative-recommendation-18-mtgr"
slug: "generative-recommendation-18-mtgr-en"
excerpt: "MTGR preserves DLRM candidate crosses, aggregates many candidates into one HSTU user sequence, and uses GLN plus dynamic masks to combine scale, quality, and causal safety."
series: "generative-recommendation"
seriesOrder: 18
draft: false
---

Generative recommendation offers a seductive promise:

> Keep a sufficiently long raw history and scale the model enough; handcrafted statistics and crosses will eventually be relearned.

[HSTU](/blog/generative-recommendation-16-hstu-en) explicitly points in that direction, and [OneRec](/blog/generative-recommendation-17-onerec-en) goes further toward replacing the funnel with one generator.

Meituan's production evidence provides a useful correction. [MTGR: Industrial-Scale Generative Recommendation Framework in Meituan](https://arxiv.org/abs/2505.18654) reports:

> Removing long-validated candidate-aware cross features causes a major quality loss that scaling the pure generative model does not recover.

MTGR does not return to a conventional DLRM. It keeps the crosses, rearranges a user and many candidates as tokens, encodes them once with HSTU, and emits CTR/CTCVR scores at candidate positions.

## The paper in 30 seconds

1. **The HSTU/OneRec limitation:** long sequences scale well, but removing compact priors such as this user's historical CTR or exposure count for the target candidate can discard highly efficient information.
2. **MTGR's answer:** aggregate K candidates from one request into one user sample; compute user features once; put candidate-specific cross features into candidate tokens; use Group-Layer Normalization and Dynamic Masking for heterogeneous semantics and temporal leakage.
3. **The evidence:** removing crosses reduces CTCVR GAUC from 0.6603 to 0.6514, below the strong DLRM's 0.6550. MTGR-large reaches 0.6646 and reports +1.90% PV_CTR and +1.02% UV_CTCVR online.

![Pure GRM versus MTGR](/blog/generative-recommendation/18-mtgr/paradigm-shift.svg)

*Figure 1. MTGR is not “traditional versus generative.” It places proven information inside scalable, reusable sequence computation.*

## What is a cross feature?

Raw inputs may contain:

```text
user: city, age, membership
history: restaurants clicked, products purchased
candidate: merchant, dish, price, delivery distance
```

Cross features directly summarize the user–candidate relationship:

```text
this user's historical CTR for this candidate category
how often this user saw this merchant
difference between usual purchase price and candidate price
number of history events matching the candidate's tags
```

They are engineered, but they compress many events into cheap target-aware priors. Under finite context and capacity, asking a model to recount one thousand events may be less efficient than providing the count.

The conventional DLRM repeats user computation:

```text
(same history, candidate 1) → score 1
(same history, candidate 2) → score 2
...
```

As candidate count grows, the large user tower runs K times.

## User-level compression with eight items

Divide inputs into:

- \(U\): static user features;
- \(S\): longer-term behavior;
- \(R\): real-time behavior from the preceding hours/day;
- \(I_i\): candidate \(i\)'s features;
- \(C_i\): user–candidate cross features.

K traditional rows become:

\[
D=[U,S,R,[C,I]_1,\ldots,[C,I]_K].
\]

In our toy catalog:

```text
shared once:
  U = city and age
  S = I01 racket, I03 climbing shoes, I05 badminton racket
  R = just saved I07 swim cap

candidate token 1:
  I06 goggles + user×swimming CTR + exposures

candidate token 2:
  I08 towel + user×recovery CTR + exposures
```

One HSTU pass emits `score(I06), score(I08), ...`. All candidates share user computation, yielding sublinear inference-cost growth with candidate count.

![User-level aggregation](/blog/generative-recommendation/18-mtgr/toy-compression.svg)

*Figure 2. Compression removes repeated computation, not user information. Every candidate keeps its own cross statistics.*

## Converting heterogeneous features into tokens

Scalar features such as age and gender become individual tokens. Every sequence item embeds and concatenates its fields, then a domain-specific MLP projects it to common width \(d_{\rm model}\).

A candidate uses \([C_i,I_i]\): crosses and item fields are fused before its token is created. Finally:

\[
\operatorname{Feat}_D
=\operatorname{Concat}
([\operatorname{Feat}_U,\operatorname{Feat}_S,
\operatorname{Feat}_R,\operatorname{Feat}_I])
\in\mathbb R^{(N_U+N_S+N_R+N_I)\times d_{\rm model}}.
\]

Candidate hidden states feed MLPs for CTR and CTCVR. MTGR does not autoregressively generate item IDs. “Generative” refers mainly to user-level sequence organization, end-to-end long-history learning, and computation reuse; the ranking output is discriminative.

## Why Group-Layer Normalization?

Age tokens, historical products, real-time events, and candidate-cross tokens come from different distributions. One normalization treatment can let one domain's scale dominate another.

MTGR normalizes by semantic group:

\[
\tilde X=\operatorname{GroupLN}(X),
\]

then projects \(Q,K,V,U\) for HSTU-style SiLU attention and gating. GLN does not make the domains semantically identical. It aligns numerical scales so attention need not learn calibration and interaction simultaneously.

![MTGR token encoding](/blog/generative-recommendation/18-mtgr/mechanism.svg)

*Figure 3. GroupLN handles distributions, Dynamic Mask handles visibility, and HSTU handles interaction. None substitutes for another.*

## Why Dynamic Masking is mandatory

Aggregating multiple events from one user over several hours puts “past” and “future relative to an earlier target” into one sequence. Full attention lets early candidates inspect later clicks, leaking labels.

MTGR defines visibility by token type:

1. **Static \(U,S\):** created before the aggregation window and visible everywhere;
2. **dynamic \(R\):** causal by timestamp;
3. **candidate tokens:** visibility depends on their time relative to real-time events, with inappropriate candidate-to-candidate paths masked.

A standard causal mask is insufficient: static user context need not be hidden, while candidates and real-time events carry distinct timestamps.

## The training system is part of the method

The TorchRec stack is modified with:

- dynamic hash tables for real-time sparse-embedding insertion/deletion;
- embedding-ID deduplication and automatic table merging;
- per-GPU dynamic batch sizes based on sequence length;
- sequence packing without cross-user attention;
- mixed precision and operator fusion.

These changes yield 1.6–2.4× training throughput over baseline TorchRec and scale beyond 100 GPUs. Without them, a 65× per-sample FLOP model would not be economical online.

## What the experiments establish

The ten-day industrial dataset contains:

| Statistic | Scale |
| --- | ---: |
| Training users | 210 million |
| Items | 4.30 million |
| Exposures | 23.74 billion |
| Clicks | 1.08 billion |
| Purchases | 180 million |

Strong DLRM `UserTower-SIM` uses 0.86 GFLOPs/example. MTGR-small, medium, and large use 5.47, 18.59, and 55.76—large is roughly 65×.

![MTGR and the cross-feature ablation](/blog/generative-recommendation/18-mtgr/evidence.svg)

*Figure 4. The truncated y-axis exposes small but operationally important GAUC differences. Removing crosses drops below the strong DLRM.*

| Model | CTR GAUC | CTCVR GAUC |
| --- | ---: | ---: |
| UserTower-SIM | 0.6792 | 0.6550 |
| MTGR-small | 0.6826 | 0.6603 |
| MTGR-medium | 0.6843 | 0.6625 |
| MTGR-large | **0.6865** | **0.6646** |
| MTGR-small w/o crosses | 0.6689 | 0.6514 |

Without GLN, CTCVR GAUC is 0.6585; without Dynamic Mask, 0.6587. Their contributions are comparable to scaling small into medium. More importantly, 0.6514 without crosses is below the DLRM's 0.6550:

> In this scenario, architectural scaling and effective features are complements, not substitutes.

A 2%-traffic online test trains MTGR on six months of data against a production DLRM optimized for two years:

| Model | PV_CTR | UV_CTCVR |
| --- | ---: | ---: |
| MTGR-small | +1.04% | +0.04% |
| MTGR-medium | +2.29% | +0.62% |
| MTGR-large | **+1.90%** | **+1.02%** |

Medium CTR exceeds large while large conversion is strongest, so multitask online metrics do not scale in lockstep. The paper states that MTGR serves main traffic, matches DLRM training cost, and reduces inference cost by 12%.

## Where MTGR breaks

### “Generative” can mislead

MTGR emits discriminative candidate logits, not item IDs or lists. It inherits sequence organization and computation reuse from GR, not TIGER/OneRec's output interface.

### Feature-engineering debt remains

Crosses require definitions, offline/online consistency, real-time computation, drift monitoring, and bias governance.

### Mask errors cause leakage

Multiple token types and windows create a much more complex mask than causal attention. Offline improvements disappear online if a path exposes future behavior; temporal replay tests are essential.

### “Scaling cannot recover it” is scenario-specific

The evidence concerns Meituan food-delivery ranking and its crosses. Better raw events, longer contexts, or different self-supervision may move the boundary elsewhere.

### History is still bounded

Long-term history is capped at 1,000 and real-time history at 100. The model is substantially larger than DLRM but does not process infinite behavior; the 65× FLOP gain depends on specialized systems.

## Why OneRec-Think comes next

HSTU, OneRec, and MTGR address scale, funnels, features, and cost, but their decision remains an implicit hidden state:

```text
the model returns I06,
but cannot naturally respond to “my shoulder hurts today; avoid intense sports,”
or expose which behaviors became which interest.
```

The LLM capability of producing text before an answer now enters industrial generative recommendation. [OneRec-Think, in the next chapter](/blog/generative-recommendation-19-onerec-think-en), aligns itemic tokens with language, trains recommendation CoT, applies GRPO, and deploys a Think-Ahead architecture.

---

**Critical takeaway:** MTGR is a healthy calibration of technological fashion. The right scaling question is not when every old feature disappears, but which information is economical for a large model to relearn and which is better supplied as an explicit statistic. Industrial optima are often recombinations, not pure replacements.
