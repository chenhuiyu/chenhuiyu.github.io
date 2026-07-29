---
title: "Treat Items as Words and Shopping Histories as Sentences | Generative Recommendation 04: BERT4Rec"
date: "2026-07-28"
updated: "2026-07-29"
category: "Generative Recommendation"
language: "en"
tags:
  - Generative Recommendation
  - Recommender Systems
  - BERT4Rec
  - Transformer
pairKey: "generative-recommendation-04-bert4rec"
slug: "generative-recommendation-04-bert4rec-en"
excerpt: "SASRec predicts only from left to right. BERT4Rec masks historical items and fills them from bidirectional context, turning an Item ID into a genuinely predictable token."
series: "generative-recommendation"
seriesOrder: 4
draft: false
---

Complete this shopping history:

```text
racket → [MASK] → shuttlecock → [MASK]
```

If the missing items are “climbing shoes” and “goggles,” evidence exists on both sides of each blank.

[SASRec, the previous paper](/blog/generative-recommendation-03-sasrec-en), has already placed Item IDs inside a Transformer: products behave like tokens, behavior order resembles a sentence, and the model predicts the next item from a prefix. Yet the full historical sequence is available during training, while the causal mask permanently hides the right side from every position.

[BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690) proposes:

> Stop training every position only to predict its successor. Randomly mask historical items and use behavior on both sides to recover them.

An Item ID is no longer merely a database key that receives a score. It becomes a token that can be masked, predicted, and defined by context.

## Understand the paper in 30 seconds

1. **The limitation left by SASRec:** left-to-right training uses only the history to the left of each position, constraining its representation and limiting the supervision drawn from one sequence.
2. **BERT4Rec’s answer:** replace a proportion $\rho$ of Item IDs with `[MASK]` and recover all hidden items with a bidirectional Transformer—the recommendation version of a Cloze task.
3. **The decisive evidence:** “BERT4Rec (1 mask)” isolates the benefit of bidirectional representation, while the multi-mask model adds richer Cloze supervision. Both steps improve results on Beauty and MovieLens-1M.

![From causal next-item prediction to bidirectional Cloze training](/blog/generative-recommendation/04-bert4rec/paradigm-shift.svg)

*Figure 1. SASRec reads only the left context during training. BERT4Rec fills blanks bidirectionally inside historical data. Online prediction still uses only behavior that has already happened.*

## What problem did the previous paper leave behind?

For

```text
I01 → I03 → I05 → I06
```

SASRec constructs supervision resembling:

```text
I01          predicts I03
I01 I03      predicts I05
I01 I03 I05  predicts I06
```

This respects online causality but creates two training issues.

### 1. Every representation is unidirectional

In a complete log, actions on both sides of `I03` may explain the interest context it occupies. A left-only representation cannot use later `I05` and `I06` to learn what `I03` means in this sequence.

### 2. Simply removing the causal mask leaks the target

A deep self-attention model cannot be made bidirectional while leaving the target Item ID in its input. Through residual or indirect attention paths, position $t$ can see the very item it is asked to predict and reduce training to copying.

BERT4Rec uses BERT’s solution: remove the target token first, then allow the remaining context to interact in both directions.

## Understand Cloze with one toy sequence

Suppose Lin’s history is

```text
I01 racket → I03 climbing shoes → I05 shuttlecock → I06 goggles
```

One corruption step selects the second and fourth positions:

```text
input:  [I01, MASK, I05, MASK]
labels: MASK₁=I03, MASK₂=I06
```

![A multi-position Cloze example for BERT4Rec](/blog/generative-recommendation/04-bert4rec/toy-cloze.svg)

*Figure 2. Only masked positions receive target losses; all visible behavior forms their bidirectional context.*

Three ideas should not be conflated:

- **mask count:** how many positions are hidden in one sequence;
- **mask proportion $\rho$:** the fraction of valid history hidden;
- **model direction:** whether a hidden position can attend to both its left and right context.

Multiple masks are more than generic augmentation. A length-$n$ sequence supplies roughly $n$ prefix targets for next-item training. Across epochs, BERT4Rec samples different sets of missing positions, exposing the model to more context–target combinations.

More masking is not automatically better. If $\rho$ is too low, supervision is sparse. If it is too high, the evidence needed to recover a target disappears. The paper tunes $\rho$ separately for each dataset.

## What happens from model input to output?

### 1. Corrupt the sequence

For user history $S^u$, sample mask positions $S_m^u$, replace them with the special `[MASK]` token, and call the result $S^{u\prime}$.

### 2. Encode bidirectionally

Item and position embeddings pass through $L$ bidirectional self-attention layers:

$$
\mathbf H^L\in\mathbb R^{n\times d}.
$$

Unlike SASRec, there is no causal mask. The final state at every hidden position can combine left and right context.

### 3. Recover only masked Item IDs

Each $\mathbf h_m^L$ enters an item-vocabulary head:

$$
P(v\mid S^{u\prime},m).
$$

### 4. Append `[MASK]` at serving time

A real recommendation request has no future actions. For next-item inference, place `[MASK]` at the end:

```text
[I01, I03, I05, MASK]
```

Read the final-position logits and select the next item. The paper also includes training samples that mask only the final position, making part of training closer to serving.

![BERT4Rec training and online inference flow](/blog/generative-recommendation/04-bert4rec/mechanism.svg)

*Figure 3. Training recovers several random internal positions. Inference recovers only the final position. Bidirectionality does not cross real time.*

For batch size $B$:

| Tensor | Shape | Meaning |
|---|---:|---|
| corrupted IDs | $[B,n]$ | histories containing several `[MASK]` tokens |
| attention | $[B,h,n,n]$ | $h$ bidirectional attention heads |
| hidden | $[B,n,d]$ | contextual state at every position |
| mask positions | $[M,2]$ | coordinates of all $M$ targets in the batch |
| item logits | $[M,|\mathcal I|]$ | item distribution for each blank |

Implementations can gather only the $M$ masked hidden states before the output projection, avoiding unnecessary logits for unsupervised positions.

## Unpack the central equation

The Cloze objective is the negative log-likelihood of all masked items:

$$
\mathcal L
=
-
\sum_{m\in S_m^u}
\log
P\left(v_m^\ast\mid S^{u\prime}\right).
$$

- $S_m^u$ is the mask set for this training instance;
- $S^{u\prime}$ is the corrupted history;
- $v_m^\ast$ is the original Item ID at position $m$;
- $P(v_m^\ast\mid S^{u\prime})$ is its recovery probability under bidirectional context.

A batch loss averages over users and masked positions.

The decisive difference from SASRec is not the softmax. It is the conditioning set:

$$
\text{SASRec:}\quad
P(v_{t+1}\mid v_{\le t}),
$$

$$
\text{BERT4Rec training:}\quad
P(v_m\mid v_{<m},v_{>m}\text{ except masks}).
$$

At inference:

$$
P(v_{n+1}\mid v_{\le n},[\mathrm{MASK}]).
$$

“Bidirectional” therefore describes the training representation. It does not give the system future events.

## Training versus inference

```text
for each user sequence S:
    choose a random mask set M with proportion rho
    S_corrupt = replace positions M with [MASK]

    hidden = bidirectional_transformer(S_corrupt)
    logits = item_head(hidden[M])
    loss = cross_entropy(logits, original_items[M])

    sometimes construct a sample that masks only the final position
```

Serving:

```text
history = keep_recent_items(user, max_length=N-1)
input = history + [[MASK]]
hidden = BERT4Rec(input)
scores = item_head(hidden[last_position])
return constrained_top_k(scores)
```

Compared with SASRec:

- training obtains supervision at several random positions;
- every masked representation uses bidirectional context;
- inference commonly re-encodes the visible window rather than updating one recurrent state;
- and the model still scores an Item ID vocabulary or candidate set at the output.

## What do the experiments actually prove?

The paper compares against POP, BPR-MF, NCF, FPMC, GRU4Rec, Caser, SASRec, and other baselines on four datasets. Against the strongest baseline, it reports average improvements of:

- **7.24% HR@10**;
- **11.03% NDCG@10**;
- **11.46% MRR**.

The most explanatory result, however, is Table 3. The authors build three stages:

1. **SASRec:** unidirectional causal attention;
2. **BERT4Rec (1 mask):** one hidden position, mainly isolating bidirectionality;
3. **BERT4Rec:** multiple masks, adding richer Cloze supervision.

![Separate gains from bidirectionality and multi-mask Cloze](/blog/generative-recommendation/04-bert4rec/evidence.svg)

*Figure 4. HR@10 values from Table 3. The two increases separately support bidirectional context and multi-position Cloze training.*

On Beauty:

$$
0.2653
\rightarrow
0.2940
\rightarrow
0.3025.
$$

On MovieLens-1M:

$$
0.6629
\rightarrow
0.6869
\rightarrow
0.6970.
$$

The first jump shows that even one masked position benefits from bidirectional context. The second shows an additional gain from the full multi-mask objective.

Other experiments indicate:

- $\rho=0.2$ beats 0.1 on all four datasets, although the tuned optimum varies by dataset;
- increasing hidden width does not guarantee continued improvement, and recurrent baselines can deteriorate at large widths;
- removing the position-wise FFN, residual path, positional embeddings, or normalization hurts performance;
- longer-sequence datasets are more sensitive to positional information and multiple heads.

## Why these numbers are not a universal leaderboard

The paper uses a common leave-one-out protocol:

- the final item of each user is test data;
- the penultimate item is validation;
- the test target is ranked with one hundred popularity-sampled unobserved items.

This enables controlled comparison but is not full-catalog retrieval. Putting a target in the Top-10 among 101 items and retrieving it from a million items are different tasks.

The protocol also treats the next logged action as the only correct answer. A user may have liked several candidates, but unclicked alternatives become negatives.

## Where does it fail?

### 1. Training and serving have different structures

An internal training mask has left and right context. The final serving mask has only left history. Final-position mask examples reduce the mismatch but do not make the two tasks identical.

### 2. Bidirectionality is not foresight

An online recommender cannot read actions that have not happened. BERT4Rec uses complete training sequences to learn better item and context representations; it does not predict with future evidence.

### 3. Computation is quadratic and awkward to update incrementally

Full self-attention costs $O(n^2d)$. A naive implementation re-encodes the window after every new event, unlike a recurrent hidden state or a decoder with an explicit KV cache.

### 4. Vocabulary still grows with the catalog

Every product is an atomic token. A million-item catalog implies large input embeddings and an enormous output matrix. A new item has no interaction-derived representation and no shared subwords to inherit.

### 5. Text and world knowledge remain outside

The model can learn that `I03` co-occurs with `I04`, but it does not know that they are “climbing shoes” and a “chalk bag.” A description such as “high-friction rubber” does not by itself ground a new item.

### 6. It restores behavior, not explanations

BERT4Rec fills in Item IDs. It does not answer “why this product?” and does not unify ratings, explanation, dialogue, or search. Those tasks still need separate heads and datasets.

## Why does the next paper need to exist?

The first act has completed four migrations:

1. **BPR:** from value fitting to relative order;
2. **GRU4Rec:** from a static user to a behavior sequence;
3. **SASRec:** from recurrent memory to attention;
4. **BERT4Rec:** from next-item prediction to masked-token learning.

Recommendation now has an NLP-like form:

```text
user history = token sequence
Item ID = vocabulary token
objective = next-token or masked-token prediction
Transformer = sequence encoder
```

But it can still say only Item IDs.

If a user asks, “Recommend equipment for indoor bouldering and explain why,” rating, sequence prediction, retrieval, and explanation may require several models. A bolder question follows:

> Can recommendation, rating, explanation, and question answering all be rewritten as language so one model learns them with a shared text-to-text objective?

Next: [P5](https://arxiv.org/abs/2203.13366)—can every recommendation task be rewritten as a sentence?

## Original paper

- Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, Peng Jiang. [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690). CIKM 2019.

[Return to the complete roadmap →](/series/generative-recommendation/en)
