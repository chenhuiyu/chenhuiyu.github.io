---
title: "What Does Generative Recommendation Actually Generate?"
date: "2026-07-24"
updated: "2026-07-29"
category: "Generative Recommendation"
language: "en"
tags:
  - Generative Recommendation
  - Recommender Systems
  - LLM
pairKey: "generative-recommendation-preface"
slug: "generative-recommendation-preface-en"
excerpt: "The same label can describe models that generate text, item IDs, full slates, or a reasoning trace. This prologue separates them precisely."
series: "generative-recommendation"
seriesOrder: 0
draft: false
---

Suppose Lin has recently clicked a tennis racket, bought a sweatband, and inspected several pairs of climbing shoes. On Friday evening, she opens an app and sees a chalk bag recommended at the top.

That recommendation could have come from four very different computations:

1. The system retrieved a few thousand candidates, scored each one, and the chalk bag happened to rank first.
2. A sequence model read her recent actions and predicted that clicking a chalk bag was the most likely next event.
3. A large language model generated the sentence, “You have been exploring indoor sports, so you may like this chalk bag.”
4. A model directly decoded the chalk bag’s **Semantic ID**, then continued decoding an entire weekend-sports slate.

All four may appear in papers described as “generative recommendation.” Yet they generate different objects, optimize different objectives, and require different decoding and serving systems.

This twenty-paper series keeps returning to one question:

> How did recommendation move from scoring available items to directly saying the answer?

## The thirty-second answer

Traditional recommendation is primarily about **evaluating candidates**. Given a user $u$ and a candidate item $i$, the model computes a matching score $s(u,i)$ and returns the Top-K candidates.

Generative recommendation is primarily about **constructing an output sequence**. Given a user history $\mathcal H_u$, the model emits $y_1,y_2,\ldots,y_T$ token by token. Those tokens may form an explanation, an item identifier, or an entire recommendation slate.

The fundamental distinction is not whether the architecture contains a Transformer or enough parameters. It is:

- what constitutes the output space;
- whether a later result conditions on earlier generated results;
- how catalog items become tokens;
- and how decoding is constrained to valid, available, policy-compliant items.

![Four output spaces for the same recommendation request](/blog/generative-recommendation/preface/generation-outputs.svg)

*Figure 1. The same request may produce candidate scores, one next-item ID, open-vocabulary language, or a constrained sequence of Semantic ID tokens.*

## 1. The old world: a multi-stage scoring machine

It is worth understanding traditional recommendation before trying to replace it. The familiar **retrieve–rank–rerank** pipeline did not arise because engineers failed to imagine end-to-end learning. It arose because a real catalog may contain millions—or far more—eligible objects.

A common production flow is:

1. **Retrieval** quickly reduces the full catalog $\mathcal I$ to hundreds or thousands of candidates.
2. **Ranking** estimates a user–item score for every candidate.
3. **Reranking** incorporates diversity, stock, policy, advertising, and other business objectives.
4. The system displays the final Top-K.

In compact form:

$$
\mathcal C_u = \operatorname{Retrieve}(\mathcal H_u), \qquad
\hat S_u = \operatorname{TopK}_{i\in\mathcal C_u} s_\theta(u,i)
$$

Here $\mathcal C_u$ is the retrieved candidate set and $s_\theta(u,i)$ is the model’s estimate of how well item $i$ matches user $u$.

This design has real advantages:

- each stage can be replaced, scaled, and debugged independently;
- popular items, user vectors, and intermediate results can be cached;
- a ranking model can consume rich cross features;
- hard constraints can be applied deterministically at the end;
- and a new item can enter an index without waiting for a generative model to memorize the entire catalog again.

The structural limitation is equally clear: **the model can only judge candidates placed in front of it**. If retrieval never puts the chalk bag in $\mathcal C_u$, no amount of ranking intelligence can recommend it.

There is also a list-level limitation. When a scorer independently evaluates

$$
s(u,I04),\quad s(u,I05),\quad s(u,I06),
$$

the three scores do not naturally express whether another chalk bag remains useful after one has already been selected. Dependencies within the slate are usually delegated to a separate reranker or a collection of rules.

Generative recommendation attempts to move some of that complexity into representation learning and decoding.

![Traditional multi-stage recommendation and a generative pipeline](/blog/generative-recommendation/preface/pipeline-shift.svg)

*Figure 2. A classical system distributes complexity across retrieval, ranking, and reranking. A generative system concentrates more of it in item representation, autoregressive decoding, and constraints. Neither is automatically superior; the optimization boundary has moved.*

## 2. “Generative recommendation” can mean five different things

The word *generation* is too broad to be useful without an output type.

| Lineage | What is mainly generated | Representative work | Role in this series |
|---|---|---|---|
| Probabilistic generative modeling | Latent variables, user representations, or interaction samples | Mult-VAE, GAN, diffusion-based recommendation | A related lineage introduced here, but not one of the twenty main papers |
| Language generation | Rating answers, explanations, preference descriptions, or dialogue | [P5](https://arxiv.org/abs/2203.13366), [M6-Rec](https://arxiv.org/abs/2205.08084) | The beginning of the LLM branch |
| Item-ID generation | An identifier that maps to one catalog item | [DSI](https://arxiv.org/abs/2202.06991), [TIGER](https://arxiv.org/abs/2305.05065) | The generative-retrieval branch |
| Slate or session generation | A dependent sequence of items or future actions | [GPTRec](https://arxiv.org/abs/2306.11114), [OneRec](https://arxiv.org/abs/2502.18965) | The industrial unification branch |
| Rationale and reasoning generation | Explanations, inferred interests, or explicit thinking traces | [OneRec-Think](https://arxiv.org/abs/2510.11639), [OneReason](https://arxiv.org/abs/2606.06260) | The final open question |

For this series, I use a narrower, testable definition:

> A generative recommender represents the recommendation target as a token sequence and directly produces item identifiers, recommendation slates, or language jointly grounded in those outputs through an autoregressive or equivalent sequence-generation process.

Its objective can be written as:

$$
P_\theta(y_{1:T}\mid \mathcal H_u)
=
\prod_{t=1}^{T}
P_\theta(y_t\mid y_{<t},\mathcal H_u)
$$

The most consequential term is not the probability. It is $y_{<t}$: output $t$ can condition on everything already generated.

The model can therefore learn more than whether swimming goggles suit Lin in isolation. It can model questions such as:

- Is a chalk bag especially coherent after climbing shoes?
- After three climbing products, should the slate introduce swimming or badminton for diversity?
- Which token prefixes can still terminate in a valid catalog item?
- Which decoding paths must be ruled out before they produce unavailable or unsafe results?

## 3. Not every sequential recommender is a generative recommender

This boundary is easy to blur.

[GRU4Rec](https://arxiv.org/abs/1511.06939), [SASRec](https://arxiv.org/abs/1808.09781), and [BERT4Rec](https://arxiv.org/abs/1904.06690) all predict items and all borrow ideas from language modeling. In common implementations, however, they still compute logits for a catalog vocabulary or a candidate set and select the highest-scoring items.

Why, then, are they the prehistory of this series?

Because they accomplished three conceptual migrations that generative recommendation depends on:

1. **A user changed from a static vector into a behavior sequence.**
2. **An Item ID changed from a database key into a learnable token.**
3. **Recommendation changed from one matching decision into conditional sequence prediction.**

Sequential recommendation and modern generative recommendation therefore form a spectrum rather than two countries separated by a clean border. Four questions locate a method more reliably than the word *generative* in its title:

- Must inference explicitly enumerate candidates?
- Is an item represented by one private token or several shared tokens?
- Does the model predict one item or jointly decode a slate or session?
- Does decoding explicitly model list dependence and catalog constraints?

## 4. The hard part: an item is not naturally a word

Language models can generate token by token because many words share a relatively stable subword vocabulary. *Climbing shoe* and *hiking shoe* are different concepts, but their tokens expose shared structure.

A product catalog behaves differently:

- every output must map back to one exact inventory object;
- catalog size may dwarf an ordinary language vocabulary;
- products are continuously added, removed, merged, and revised;
- textually similar products may attract different user populations;
- textually unrelated products may be strongly connected by co-consumption.

Item tokenization is therefore not a minor preprocessing decision. It is one of the central modeling choices in generative recommendation.

### Atomic IDs: strong memorization

The simplest design allocates one indivisible token per item: the climbing shoes are `<I03>`.

The mapping is unique and executable, but vocabulary size grows linearly with the catalog. Nothing in the symbols `<I03>` and `<I04>` tells the model that they mean climbing shoes and a chalk bag, much less that users often purchase them together.

### Textual IDs: strong semantics

Another design represents the item with its title or attributes, such as “indoor climbing shoes.”

This can reuse world knowledge already present in a language model and can help with new items. The cost is that names are not guaranteed to be unique, stable, or executable. A fluent description does not prove that the exact product exists in the current catalog.

### Semantic IDs: a compromise between memory and generalization

A Semantic ID quantizes a continuous item representation into a short code sequence:

```text
<sport> <climb> <shoe>
```

Real codes are rarely this human-readable, but the intuition is useful: multiple items can share prefixes or codewords while decoding progressively narrows the search space. TIGER uses an RQ-VAE to construct such identifiers. LC-Rec, LETTER, and ETEGRec then ask how much textual semantics and collaborative structure the codes should carry, and whether the tokenizer should be optimized jointly with recommendation.

This representation is not free. Quantization loses information, items can collide, code usage can collapse toward a small subset, and a changing catalog creates an indexing-maintenance problem.

![Atomic, textual, and semantic item identifiers](/blog/generative-recommendation/preface/item-identifiers.svg)

*Figure 3. Atomic IDs favor exact memorization, textual IDs favor open-world semantics, and Semantic IDs seek a balance among executability, shared structure, and catalog scale.*

## 5. Why Transformers and LLMs accelerated the shift

Transformers entered recommendation for more than their ability to chat. They provide a reusable sequence machine:

- user behavior can be expressed as an input sequence;
- items can be represented as tokens;
- tasks can be distinguished with prompts;
- recommendation outputs can be sequences;
- training can use masked prediction or next-token prediction;
- and model size, context length, and data volume can be scaled within one framework.

But recommendation is not language modeling with a different vocabulary.

At least four conflicts remain:

1. **Language semantics are not collaborative semantics.** Descriptively similar items are not necessarily preferred by the same users.
2. **Long context is not useful history.** Fitting years of behavior into a context window does not guarantee that the model can find what matters now.
3. **An open vocabulary is not a valid catalog.** Language may be generated freely; a recommended item must exist, be available, and satisfy policy.
4. **Next-token likelihood is not platform value.** Clicks, watch time, diversity, satisfaction, and long-term retention can disagree.

The LLM4Rec branch therefore spends much of its effort on alignment: connecting a language model’s world knowledge with collaborative knowledge learned from co-occurrence, clicks, and purchases.

The Semantic ID branch performs a different alignment: making item tokens simultaneously useful for compression, retrieval, cold start, and the downstream recommendation objective.

## 6. Two branches converge in industrial systems

The twenty papers are organized by technical conflict rather than strict publication date because the field visibly splits after 2022.

The **LLM branch** begins with P5 and M6-Rec, then encounters the limits of generic language models, long behavioral contexts, and separated language and collaborative spaces.

The **generative-retrieval branch** begins with DSI and TIGER and concentrates on Item IDs, Semantic IDs, codebooks, slate generation, and end-to-end tokenization.

By [HSTU](https://arxiv.org/abs/2402.17152), OneRec, and [MTGR](https://arxiv.org/abs/2505.18654), the central questions are no longer confined to Recall@K on an academic benchmark:

- Can one sequence model serve real industrial traffic?
- Can it actually absorb retrieval, ranking, and reranking?
- Should carefully engineered cross features be discarded?
- Can training and inference cost scale with model capacity?
- Can user feedback enter the objective through DPO or reinforcement learning?

OneRec-Think and OneReason then add explicit reasoning. Yet producing a plausible rationale and faithfully exposing why a model selected an item are different achievements. This series will not assume that reasoning helps; it will treat that claim as a hypothesis to test.

![The five-act roadmap for the twenty-paper series](/blog/generative-recommendation/preface/series-roadmap.svg)

*Figure 4. The LLM and Semantic ID branches develop in parallel, then meet again in industrial unification, preference alignment, and recommendation reasoning.*

## 7. How we will read the twenty papers

Every article will reuse the same toy user, Lin, and the same eight sports items. The example remains stable while the model’s interpretation changes.

Each article has three layers:

1. **Beginner layer:** the limitation left by the previous method, explained without requiring the paper.
2. **Engineering layer:** concrete inputs, outputs, tensor shapes, training examples, sampling or decoding, and complexity.
3. **Paper layer:** one central equation, one decisive experiment, one important ablation, and what the paper did not prove.

The notation will also remain stable:

| Symbol | Meaning |
|---|---|
| $u$ | user |
| $i$ | item |
| $\mathcal I$ | full item catalog |
| $\mathcal H_u$ | user behavior history |
| $s(u,i)$ | user–item matching score |
| $z_i=(z_i^1,\ldots,z_i^m)$ | a multi-token Semantic ID |
| $S_u$ | a recommendation slate or session |

The goal is not to memorize twenty acronyms. After every paper, the reader should be able to answer:

> Which assumption about recommendation did this paper change, and which unsolved problem did it hand to the next paper?

## 8. There is no ending yet

Generative recommendation still has unresolved problems:

- How can a generative index learn catalog updates without catastrophic forgetting?
- Should Semantic IDs preserve content semantics or collaborative signals first?
- Can beam search and constrained decoding meet strict online latency?
- Does autoregressively generating a slate beat a strong retriever plus reranker under a fair budget?
- With exposure-biased clicks, is the model learning preference or merely imitating the previous recommender?
- Do multi-objective rewards improve long-term experience or create new proxy metrics?
- How can a recommendation chain of thought be tested for faithfulness rather than readability?

This series is therefore not a victory story in which a new paradigm inevitably replaces an old one.

Its real value is that generative recommendation forces the field to reopen three old questions:

1. How should a user’s preference be represented?
2. How should millions of items be organized?
3. Are we optimizing one click or a longer, more coherent experience?

Before asking a model to generate an item, we need to return to the original difficulty: users never explicitly tell the system what they dislike. The data contains clicks, purchases, and silence.

The next article begins with [BPR](https://arxiv.org/abs/1205.2618) and the first modest skill a recommender had to learn:

> For one user, which item should rank above which?

[Explore the complete twenty-paper roadmap →](/series/generative-recommendation/en)

## Original papers referenced

- [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618)
- [Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939)
- [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)
- [BERT4Rec](https://arxiv.org/abs/1904.06690)
- [P5: Recommendation as Language Processing](https://arxiv.org/abs/2203.13366)
- [M6-Rec](https://arxiv.org/abs/2205.08084)
- [Transformer Memory as a Differentiable Search Index](https://arxiv.org/abs/2202.06991)
- [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)
- [Generative Sequential Recommendation with GPTRec](https://arxiv.org/abs/2306.11114)
- [Actions Speak Louder than Words: HSTU](https://arxiv.org/abs/2402.17152)
- [OneRec](https://arxiv.org/abs/2502.18965)
- [MTGR](https://arxiv.org/abs/2505.18654)
- [OneRec-Think](https://arxiv.org/abs/2510.11639)
- [OneReason](https://arxiv.org/abs/2606.06260)
