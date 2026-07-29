---
title: "Can One Model Retrieve and Rank in a Single Pass? | Generative Recommendation 17: OneRec"
date: "2026-07-29"
updated: "2026-07-29"
category: "Generative Recommendation"
language: "en"
tags:
  - Generative Recommendation
  - Recommender Systems
  - OneRec
  - Preference Alignment
pairKey: "generative-recommendation-17-onerec"
slug: "generative-recommendation-17-onerec-en"
excerpt: "OneRec combines balanced Semantic IDs, session-wise generation, Sparse MoE, and iterative DPO to replace Kuaishou's retrieve–prerank–rank funnel with one list generator."
series: "generative-recommendation"
seriesOrder: 17
draft: false
---

A conventional recommendation funnel is a relay:

```text
millions of items
  ↓ retrieval keeps thousands
  ↓ pre-ranking keeps hundreds
  ↓ ranking orders the final list
```

Each stage saves compute and sets the ceiling for the next. A missed retrieval candidate can never be rescued by a perfect ranker. Stages optimize clicks, watches, and interactions separately, while rules assemble the final list.

[HSTU](/blog/generative-recommendation-16-hstu-en) puts ranking and retrieval into one sequential-transduction framework, but at different positions with different outputs. [OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment](https://arxiv.org/abs/2502.18965) proposes a more radical industrial design:

> Generate an entire high-value session from user history with one model instead of filtering through independent rankers.

## The paper in 30 seconds

1. **HSTU's open problem:** a shared backbone is not a shared final decision; the cascade still has retrieval ceilings and disconnected objectives.
2. **OneRec's answer:** build three-level IDs with balanced residual k-means; generate 5–10-video sessions with a T5 encoder–decoder; scale the decoder with Sparse MoE; and iteratively DPO-train preference pairs mined from its own beam outputs by a reward model.
3. **The evidence:** maximum session-watch-time rises from TIGER-1B's 0.1368 to OneRec-1B's 0.1529 and IPA's 0.1933. Online, IPA improves total watch time by 1.68% and average view duration by 6.56% over the production multi-stage system.

![From cascade to session generation](/blog/generative-recommendation/17-onerec/paradigm-shift.svg)

*Figure 1. “Single-stage” means one model emits a displayable list. Tokenization, beam search, rewards, and online training still surround it.*

## Why next-item prediction is insufficient

TIGER-style training learns:

```text
history → one next item
```

To serve ten videos, a system can run independent generation and then deduplicate, diversify, and interleave creators with handcrafted rules. The model never observes the list as a unit:

- Is item two redundant with item one?
- Are the first three all emotionally heavy?
- Where should a new interest enter?
- How should total session watch value be traded against per-item relevance?

OneRec defines a session as the 5–10 short videos returned for one request. A high-value training session satisfies criteria such as:

1. at least five videos actually watched;
2. total duration above a threshold;
3. likes, saves, or shares.

The model learns from log-filtered “good lists,” not arbitrary exposure lists.

## Session-wise generation with eight items

Given:

```text
I01 tennis racket → I03 climbing shoes
→ I05 badminton racket → I07 swim cap
```

the target becomes:

```text
I06 goggles → I08 sports towel → I04 chalk bag → I02 tennis balls
```

When generating I08, the decoder sees I06. Both item order and code order enter:

\[
P(S\mid H_u)
=\prod_{i=1}^{m}\prod_{j=1}^{L}
P(s_{i,j}\mid H_u,S_{<i},s_{i,<j}).
\]

\(m\) is session length, \(L\) is ID length, \(S_{<i}\) is the generated item prefix, and \(s_{i,<j}\) is the current item's code prefix.

![Session-wise generation in the toy catalog](/blog/generative-recommendation/17-onerec/toy-session.svg)

*Figure 2. Session-wise factorization provides list dependencies, while the data definition decides what “high value” means.*

## Balancing the item address space

OneRec starts from multimodal video embeddings and applies multilevel residual k-means rather than RQ-VAE. Ordinary clustering creates an hourglass: some codes contain too many videos and others are nearly empty, so early beam branches are dominated by popular codes.

Balanced k-means fixes cluster capacity:

\[
w=\frac{|V|}{K}.
\]

For each centroid, assign its nearest \(w\) still-unallocated videos and recalculate the center until assignments converge. Each level quantizes the preceding residual:

\[
s_i^l=\arg\min_k\lVert r_i^l-c_k^l\rVert_2^2,\qquad
r_i^{l+1}=r_i^l-c_{s_i^l}^l.
\]

Distance preserves similarity; fixed capacity enforces code usage. Unlike LC-Rec's Sinkhorn soft assignment, this is a sequential hard-capacity clustering algorithm.

## Scaling with Sparse MoE

The encoder models positively watched and interacted histories. The decoder autoregressively generates the session.

Making every decoder FFN larger increases FLOPs per token. OneRec replaces FFNs with Sparse Mixture-of-Experts:

\[
h_{t}^{l+1}
=h_t^l+\sum_{i=1}^{N_{\rm MoE}}g_{i,t}\operatorname{FFN}_i(h_t^l),
\]

where only router top-\(K_{\rm MoE}\) gates are nonzero. The industrial configuration has 24 experts and activates two per token. A 1B-parameter model uses roughly 13% of parameters during inference.

![The complete OneRec loop](/blog/generative-recommendation/17-onerec/mechanism.svg)

*Figure 3. Tokenization, session likelihood, and preference alignment optimize different layers. MoE supplies capacity; it does not define a good session.*

## Why recommendation DPO needs synthetic comparisons

Language DPO often has explicit human comparisons. A recommendation request displays only one list; the user never consumes its counterfactual alternative, so natural chosen/rejected pairs do not exist.

OneRec first trains a session reward model \(R(u,S)\). User–item target-aware representations interact within the session and feed separate heads for:

- `swt`: session watch time;
- `vtr`: view probability;
- `wtr`: follow probability;
- `ltr`: like probability.

The model uses multitask BCE on logged outcomes. Iterative Preference Alignment then runs:

```text
beam-search N=128 sessions from the current OneRec
score every session with the reward model
highest score → chosen S_w
lowest score  → rejected S_l
update with NTP + λDPO
snapshot the new model and mine new hard negatives
```

\[
\mathcal L_{\rm DPO}
=-\log\sigma\left[
\beta\log\frac{\pi_\theta(S_w\mid H)}{\pi_{\rm ref}(S_w\mid H)}
-\beta\log\frac{\pi_\theta(S_l\mid H)}{\pi_{\rm ref}(S_l\mid H)}
\right].
\]

Generating 128 alternatives is expensive, so only \(r_{\rm DPO}=1\%\) of training samples use IPA. Raising the ratio to 5% costs about five times the sampling GPUs with marginal gain; 1% retains roughly 95% of the maximum observed effect.

## What the experiments establish

Offline evaluation uses reward-model estimates, not conventional Recall/NDCG:

![TIGER, OneRec, and IPA](/blog/generative-recommendation/17-onerec/evidence.svg)

*Figure 4. Session generation improves over point-wise TIGER, and IPA substantially raises the best reward-model-scored candidate.*

| Model | max swt | max ltr |
| --- | ---: | ---: |
| TIGER-1B | 0.1368 | 0.0579 |
| OneRec-1B | 0.1529 | 0.0660 |
| OneRec-1B + IPA | **0.1933** | **0.1203** |

The prose reports +4.04% max swt and +5.43% max ltr for IPA over OneRec-1B under its comparison protocol. Those relative changes should not be recomputed directly from the displayed aggregate values, which combine different statistics and repeated evaluations.

Scaling from 0.05B to 0.1B yields a maximum 14.45% accuracy gain; moving to 0.2B, 0.5B, and 1B adds 5.09%, 5.70%, and 5.69%.

An online test on 1% of main traffic reports:

| Model | Total watch time | Average view duration |
| --- | ---: | ---: |
| OneRec-0.1B | +0.57% | +4.26% |
| OneRec-1B | +1.21% | +5.01% |
| OneRec-1B + IPA | **+1.68%** | **+6.56%** |

This is stronger evidence than reward-model scores: in Kuaishou's principal short-video scenario, a generated session beats the mature cascade baseline.

## Where OneRec breaks

### Preference means reward-model preference

Neither alternative list is shown to the user. The RM chooses among OneRec's own candidates. DPO amplifies RM bias and may learn reward-model exploits rather than long-term satisfaction.

### High-value sessions are selected data

Filtering for long watches and interactions favors active users, popular content, and the incumbent policy. A skipped video may be irrelevant, position-biased, or simply shown when the user left.

### Single-stage does not mean single-component

The system still requires multimodal encoding, balanced codebooks, beam search, 24 experts, a reward model, a DPO sampling service, incremental training, and code mapping. It removes ranker stages, not all system complexity.

### List errors compound

An early wrong ID becomes later context. Beam search mitigates the problem at a latency cost. Balanced codes make the tree traversable but may split natural content clusters.

### External validation is limited

Data and metrics are private, and the arXiv manuscript still contains uncleaned conference-template placeholders. The online result matters, but outsiders cannot reproduce it or assess long-term stability.

## Why MTGR comes next

OneRec argues for a generator replacing the cascade. A related HSTU claim is that sufficiently long raw histories and sufficiently large models can replace traditional hand-engineered statistics and cross features.

Meituan reports the opposite observation:

> Removing candidate-aware cross features causes a major quality loss that scaling the generative model does not recover.

[MTGR, in the next chapter](/blog/generative-recommendation-18-mtgr-en), keeps the scaling benefits but adopts a pragmatic hybrid: preserve proven DLRM features, aggregate many candidates into one user's token sequence, encode it once with HSTU, and emit discriminative scores.

---

**Critical takeaway:** OneRec turns the final display session into an industrial generation target and supplies online evidence that funnel unification can work. It does not remove recommendation's intermediate machinery; it moves complexity from multiple rankers into the item language, generation procedure, and preference-alignment loop.
