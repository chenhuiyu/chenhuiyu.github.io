---
title: "What Does Generative Recommendation Actually Generate?"
date: "2026-07-24"
updated: "2026-07-24"
category: "Generative Recommendation"
language: "en"
tags:
  - Generative Recommendation
  - Recommender Systems
  - LLM
pairKey: "generative-recommendation-preface"
slug: "generative-recommendation-preface-en"
excerpt: "Before reading 20 papers, distinguish whether a recommender is generating language, item IDs, an entire list, or something that merely looks like reasoning."
series: "generative-recommendation"
seriesOrder: 0
draft: false
---

Recommender systems did not begin by “answering” anything.

They behaved more like quiet scoring machines. Given a user $u$ and an item
$i$, the model learned a score $s(u,i)$ and ranked the items with the largest
scores first.

$$
\hat{i} = \arg\max_i s(u,i)
$$

Generative recommenders now appear to do something fundamentally different.
They can decode item IDs token by token, write an entire recommendation list,
explain a choice, and even produce an explicit “thinking” trace before the
answer.

This series keeps returning to one question:

> How did recommendation move from scoring every item to directly saying the answer?

## The same name can describe different lineages

VAEs, GANs, and diffusion models have long been used in recommendation, and
that work is also called “generative recommendation.” Those models may generate
latent variables, user representations, or samples from a learned
distribution, but they are not the direct lineage followed here.

This series focuses on a more specific transformation: **recasting
recommendation as autoregressive generation**. Instead of only evaluating
candidate scores, the model learns
$P(\text{output tokens}\mid\text{user history})$.

The decisive question is not whether the model generates, but what it
generates:

- **Language answers:** ratings, explanations, and preference decisions become text-to-text tasks.
- **Item identifiers:** each Item ID becomes a sequence of decodable tokens.
- **Complete lists:** every generated recommendation can depend on earlier items in the list.
- **Reasons and reasoning traces:** the model attempts to express both what to choose and why.

## The story begins with a scoring machine

The first four papers form the prehistory.

BPR establishes pairwise ranking for implicit feedback. GRU4Rec recognizes
that a user is not a static vector but a sequence of actions. SASRec uses
attention to find the history that matters for the next action. BERT4Rec then
treats Item IDs as tokens and behavioral histories as sentences.

Recommendation is not yet generating items, but language-model-style
representations have entered the field.

## Two branches separate

The first is the **LLM branch**.

P5 casts multiple recommendation tasks as text-to-text. M6-Rec pushes that
unification toward an industrial foundation model. TALLRec, ReLLa, and LLaRA
then expose the limits: a general-purpose LLM can be fluent without
understanding collaborative preference, and a history fitting inside the
context window does not mean the model knows which actions matter.

The second is the **Semantic ID and generative retrieval branch**.

DSI first shows that retrieval can generate document IDs instead of looking
them up. TIGER, GPTRec, LC-Rec, LETTER, and ETEGRec move the problem toward the
central component of a generative recommender: an item tokenizer whose codes
capture semantics and collaborative behavior without collapsing its codebook.

## The branches converge in industrial systems

HSTU rewrites long user histories as sequential transduction. OneRec attempts
to unify retrieval and ranking in one model. MTGR reminds us that an elegant
generative formulation still faces cross features, latency, and cost.
OneRec-Think and OneReason then push the question into more provocative
territory:

> If a recommender looks as though it is reasoning, does it actually understand preference?

There is no settled answer. That open problem is where this series is heading.

## How the 20 papers will be read

Every article has three layers:

1. **Beginner layer:** one recurring toy user and eight items reveal the problem left by the previous model.
2. **Engineering layer:** inputs, outputs, tensor shapes, training data, inference steps, and complexity.
3. **Paper layer:** one central equation, one decisive experiment, one important ablation, and what the paper did not prove.

The recurring toy sequence is:

```text
tennis → climbing → badminton → swimming
```

The same history will pass through a scoring model, an RNN, a Transformer, an
LLM, a Semantic ID tokenizer, and finally a generative recommender.

Next, we return to the beginning: **BPR, and why recommendation originally only
needed to learn which item should rank above another.**

[Explore the complete 20-paper roadmap →](/series/generative-recommendation/en)
