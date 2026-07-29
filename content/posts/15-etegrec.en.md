---
title: "The Tokenizer Should Not Be Merely Offline Preprocessing | Generative Recommendation 15: ETEGRec"
date: "2026-07-29"
updated: "2026-07-29"
category: "Generative Recommendation"
language: "en"
tags:
  - Generative Recommendation
  - Recommender Systems
  - ETEGRec
  - End-to-End Tokenization
pairKey: "generative-recommendation-15-etegrec"
slug: "generative-recommendation-15-etegrec-en"
excerpt: "ETEGRec connects the tokenizer and generator with sequence–item and preference–semantic alignment, then alternates optimization so item IDs and user states co-evolve."
series: "generative-recommendation"
seriesOrder: 15
draft: false
---

The previous four papers share one pipeline:

```text
train tokenizer → freeze item IDs → train recommender
```

It resembles a lexicographer inventing a vocabulary before an author writes a novel. If a word is difficult to predict from context, the author cannot request a better word; if the author's context changes, the dictionary cannot see it.

[LETTER](/blog/generative-recommendation-14-letter-en) lets the tokenizer observe conventional collaborative embeddings, but the final generator still accepts a fixed address system. [ETEGRec: End-to-End Learnable Item Tokenization for Generative Recommendation](https://arxiv.org/abs/2409.05546) makes the two models align:

- the history encoder and target item should predict the same code distributions;
- the decoder's user-preference state should be close to the target item's reconstructed semantics;
- tokenizer and recommender alternate updates and co-evolve.

## The paper in 30 seconds

1. **LETTER's open problem:** the tokenizer absorbs collaborative signals but never sees how the final generator represents history and preference; once frozen, recommendation loss cannot alter item codes.
2. **ETEGRec's answer:** SIA aligns the sequence and target item at the discrete code-distribution level; PSA aligns decoder preference and target item in continuous space; each cycle updates the tokenizer and then the recommender.
3. **The result:** on Game, ETEGRec reaches 0.0507 NDCG@10 versus LETTER's 0.0475. Removing alternating training drops it to 0.0428, a larger loss than removing either alignment term alone.

“End-to-end” needs a precise qualifier:

> ETEGRec does not send one gradient through a discrete argmin from generation loss directly into the codebook. It achieves co-evolution through shared alignment objectives and alternating frozen updates. It is objective-coupled end to end, not one fully differentiable computation graph.

![From a frozen dictionary to co-evolution](/blog/generative-recommendation/15-etegrec/paradigm-shift.svg)

*Figure 1. Intermediate representations supervise both sides, while each update phase freezes the other side for stability.*

## Two kinds of agreement in eight items

The target is `I06 goggles`, after:

```text
I01 tennis racket → I03 climbing shoes
→ I05 badminton racket → I07 swim cap
```

The tokenizer receives I06's collaborative embedding \(z\), quantizes it as `〈4,2,1〉`, and reconstructs \(\tilde z\).

The recommender has two views:

1. the encoder pools history into \(z^E\), asking which item codes should follow;
2. the decoder's first state \(h^D\) summarizes the current preference, asking what kind of item fits this user now.

ETEGRec enforces:

```text
SIA: z^E and target z agree at every code level
PSA: h^D and target reconstruction z̃ are close in continuous space
```

The first aligns addresses; the second aligns meanings.

![Dual alignment in the toy catalog](/blog/generative-recommendation/15-etegrec/toy-coevolution.svg)

*Figure 2. SIA links the encoder to the quantizer; PSA links the decoder to reconstructed item semantics.*

## SIA: agreement over code distributions

Mean-pooling history states produces \(z^E\). The item tokenizer receives the target's collaborative representation \(z\). Passing both through the same residual quantizer yields soft distributions at level \(l\):

\[
p_l(k\mid z),\qquad p_l(k\mid z^E).
\]

Sequence–Item Alignment minimizes symmetric KL at every level:

\[
\mathcal L_{\text{SIA}}
=\sum_l
\Big[
D_{\mathrm{KL}}(p_l(\cdot\mid z)\Vert p_l(\cdot\mid z^E))
+D_{\mathrm{KL}}(p_l(\cdot\mid z^E)\Vert p_l(\cdot\mid z))
\Big].
\]

The paper's displayed equation appears to place a leading minus sign before the KL sum; its described role is to minimize distribution disagreement, which is the convention used here.

If the tokenizer assigns target code 4 probability 0.9 while history favors code 7, SIA encourages both a more history-predictable code boundary and an encoder state closer to the target address.

## PSA: preference meets item semantics

Preference–Semantic Alignment uses decoder state \(h^D\) and target reconstruction \(\tilde z\). Matching examples are positives and other batch targets are negatives in bidirectional InfoNCE:

\[
\mathcal L_{\text{PSA}}
=\frac12(\ell_{h\to z}+\ell_{z\to h}),
\]

with one direction:

\[
\ell_{h\to z}
=-\sum_i\log
\frac{\exp(\operatorname{sim}(h_i^D,\tilde z_i)/\tau)}
{\sum_j\exp(\operatorname{sim}(h_i^D,\tilde z_j)/\tau)}.
\]

SIA is level-wise discrete agreement; PSA is global continuous similarity. They supervise different interfaces.

## From input to output

Experiments first train SASRec to obtain 256-dimensional collaborative item embeddings, then initialize an RQ-VAE with them. The tokenizer uses three levels of 256 codes plus a collision token. The generator is a T5-like encoder–decoder.

![ETEGRec's dual alignment](/blog/generative-recommendation/15-etegrec/mechanism.svg)

*Figure 3. Collaborative item vectors, encoder sequence states, and decoder preference states meet through two objectives.*

The two subsystem losses are:

\[
\mathcal L_{\mathrm{IT}}
=\mathcal L_{\mathrm{SQ}}
+\mu\mathcal L_{\mathrm{SIA}}
+\lambda\mathcal L_{\mathrm{PSA}},
\]

\[
\mathcal L_{\mathrm{GR}}
=\mathcal L_{\mathrm{REC}}
+\mu\mathcal L_{\mathrm{SIA}}
+\lambda\mathcal L_{\mathrm{PSA}}.
\]

\(\mathcal L_{\mathrm{SQ}}\) trains semantic quantization; \(\mathcal L_{\mathrm{REC}}\) trains next-item generation. The alignment terms become a shared language.

## Why not update everything simultaneously?

Hard item codes can jump. If the tokenizer changes its codebook every step while the generator learns the same labels, the target continually moves: yesterday's `〈4,2,1〉` may denote a different item today.

ETEGRec alternates:

```text
repeat cycle:
    freeze recommender
    update tokenizer with L_IT for 1 epoch

    freeze tokenizer
    update recommender with L_GR for C−1 epochs

after tokenizer convergence:
    freeze tokenizer permanently
    finish recommender training
```

The schedule acknowledges the conflict between co-evolution and label stability: improve the address early, then lock the vocabulary late.

## What the experiments establish

The three datasets contain roughly 24.5K–25.8K items. Evaluation is full ranking and maximum sequence length is 50.

![ETEGRec, LETTER, and no alternating training](/blog/generative-recommendation/15-etegrec/evidence.svg)

*Figure 4. The full model consistently beats LETTER; removing alternating training falls below LETTER, so coupled objectives require a stable update schedule.*

| Dataset | LETTER | ETEGRec | w/o alternating |
| --- | ---: | ---: | ---: |
| Instrument | 0.0310 | **0.0331** | 0.0277 |
| Scientific | 0.0230 | **0.0241** | 0.0198 |
| Game | 0.0475 | **0.0507** | 0.0428 |

Removing both alignments yields 0.0317, 0.0224, and 0.0478—still above the non-alternating variant. The evidence therefore supports two layers:

1. SIA and PSA add useful information;
2. **the optimization schedule matters more than merely adding those losses.**

The paper reports \(p<0.01\) significance against the strongest baseline across metrics. Absolute improvements remain modest—for Game, 0.0475 to 0.0507. The work validates that a tokenizer should observe downstream states; it does not close the industrial problem.

## Where ETEGRec breaks

### “End-to-end” is still an alternating approximation

Hard argmin quantization is not differentiable, and the two modules do not update through one backward pass. The system must manage frozen phases, cycle length, and convergence.

### Co-evolution is inherently unstable

The paper's own ablation shows severe degradation without alternating updates. Restarting co-evolution after an online catalog shift can move every address again.

### The tokenizer is ultimately frozen

The method begins from the limitation of static IDs, yet eventually returns to static IDs because a serving system requires stable mappings. That is evidence of an unresolved deployment constraint, not a logical contradiction.

### Training and versioning are complex

Initialize SASRec, train RQ-VAE, run multiple cycles, rewrite histories, and finish generator training. Model, codebook, trie, and cache versions must remain consistent throughout.

### Industrial scale remains open

Roughly 25K items and offline full ranking do not answer beam latency, streaming insertion and deletion, cross-domain behavior, or real-time feedback at millions of items.

## Why HSTU comes next

The Semantic-ID branch now forms a compact arc:

```text
DSI: IDs can be generated
TIGER: content can be quantized into item IDs
LC-Rec: IDs can connect LLMs and collaborative tasks
LETTER: tokenizers can optimize collaboration and balance
ETEGRec: tokenizers can observe downstream recommendation states
```

Most evidence still comes from small catalogs, offline datasets, and encoder–decoder beam search. Industrial systems need architectures that process huge interaction corpora and long histories efficiently, unify retrieval and ranking without exploding latency, and absorb policy or reasoning feedback.

[HSTU, in the next chapter](/blog/generative-recommendation-16-hstu-en), shifts the lens from item vocabulary to an industrial generative-recommendation backbone, redesigning attention, time, and feature interaction for scale.

---

**Critical takeaway:** ETEGRec's most important finding may be its own limitation. An item vocabulary must change to adapt downstream, while an online system must freeze it for stable identity. Alternating optimization is a practical bridge between those requirements, not yet their final resolution.
