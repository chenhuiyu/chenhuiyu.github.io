---
title: "HSTU starts with a question: which sequence are we predicting?"
date: "2026-09-10"
updated: "2026-09-10"
category: "HSTU"
language: "en"
tags: ["hstu", "Hands-on", "Model Learning"]
pairKey: "hstu-sequence"
slug: "hstu-sequence-en"
excerpt: "Separate content, actions, targets and candidates; understand sequential transduction and leakage boundaries."
series: "hstu"
seriesOrder: 1
draft: false
---

![HSTU starts with a question: which sequence are we predicting?](/learning/hstu-sequence-en.svg)

## A complete explanation from first principles

### Define the event being predicted

Recommendation may predict impressions, clicks, dwell, saves or purchases. These targets have different labels, delays and biases. Before building a sequence, specify the decision time, available history, candidate set and target event. A sophisticated sequence model can still optimize the wrong task if those definitions are missing.

Recommendation tokens may encode items, actions, time and context rather than text vocabulary entries. An item ID is an identifier, not automatically a semantic or visual representation. Its meaning must remain consistent with item state and feature versions.

### Construct a sample from a small event log

Suppose a user views A, clicks B and buys C. Next-item prediction can use `[A]` to predict B and `[A,B]` to predict C. Predicting purchase after exposure additionally requires exposed candidates and non-purchase outcomes. Every absent item is not an observed dislike.

Action types and elapsed time matter. Repeated browsing without buying differs from a completed purchase; yesterday's event differs from one a year ago. A sequence architecture can express these distinctions only if input construction and supervision preserve them.

### Prevent future information from entering features

A sample at time t may use only information available then. Later title updates, future purchases and future popularity aggregates can leak the answer. A correct attention mask cannot remove leakage already introduced during feature generation.

Random interaction splits can distribute near-duplicate histories across train and test. Simulate deployment with temporal splits and state whether users or items may have appeared during training. Report new-user, new-item and established-entity slices separately.

### Align padding and labels

Dense batches pad histories to a shared length. Padding is storage, not an item to supervise or recommend. Next-step training shifts input positions against targets and excludes invalid positions from the appropriate attention, loss and ranking operations.

The browser example uses real item IDs zero, one and two with no padding. The full PyTorch notebook uses a separate padding convention. Mixing conventions can accidentally remove a real item or recommend a padding token.

### Negative sampling changes evaluation difficulty

Selecting the correct item against a hundred random negatives is not the same as retrieving it from a million-item catalog. Sampled HR@10 and full-catalog HR@10 measure different tasks. The sampling distribution also changes which errors receive training pressure, while observed popularity reflects exposure as well as preference.

Record candidates, negative sampling, duplicate handling and filtering. Previously viewed items can remain relevant in some products; filtering them is a task decision, not a universal rule.

The browser notebook predicts the next item in a three-item synthetic cycle. Fast learning verifies computation and gradients, not understanding of real users. Shuffle the cycle, shorten history or alter targets to test what information the model uses. Add real time, actions and content features individually so gains remain attributable.

## Do not start by replacing a Transformer

Users see content and produce clicks, skips, watch time or purchases. HSTU's generative-recommendation setting models this temporal process. Generative does not necessarily mean natural-language output or Semantic-ID decoding. Retrieval can predict the next content representation; ranking can predict an action conditioned on a candidate.

This path extends the existing [HSTU overview](/blog/generative-recommendation-16-hstu-en) with data and tensor exercises.

## Respect the supervision boundary

Suppose history contains a clicked tennis item at t1 and a long-watched climbing item at t2. At t3, a swimming-video candidate arrives. Predicting its action may use earlier observed actions and current candidate content, but not the completed t3 watch time. Future-updated timestamps, aggregates and dwell statistics can leak labels.

A teaching next-item task uses `items[:-1]` to predict `items[1:]`. That is not identical to industrial interleaved content/action inputs. The notebook explicitly chooses the simpler task for CPU training and does not claim to reproduce an industrial system.

## From lists to batches

```python
import torch
sequences = [[1, 3, 4, 2], [2, 4, 1]]
inputs = torch.tensor([[1,3,4], [2,4,0]])
targets = torch.tensor([[3,4,2], [4,1,0]])
valid = inputs.ne(0)
print(inputs.shape, valid.sum(1))
```

Zero is reserved for padding and must not enter ordinary recommendation top-k. Real features can include items, creators, action types, time and content embeddings. Document update timestamps and missing-value handling. Concatenating more untrained features does not automatically improve the system.

## Define candidates before metrics

Full-catalog retrieval differs from ranking one positive among 100 sampled negatives. Negative sampling distribution affects difficulty and popularity bias. State whether previously seen items are filtered; repeat-consumption tasks may require retaining them. Splits must respect global time and feature availability, not only each user's last interaction.

## Check your understanding

**Question:** Does using only earlier actions guarantee no leakage?

<details><summary>Answer</summary>No. Item embeddings, popularity statistics, user aggregates and candidate sets can still incorporate future data. Check point-in-time availability for every field.</details>

## Primary sources and further reading

- [HSTU original paper](https://arxiv.org/abs/2402.17152)
- [Official generative-recommenders implementation](https://github.com/meta-recsys/generative-recommenders)
- [SASRec baseline](https://arxiv.org/abs/1808.09781)

Sources checked on 2026-09-10. Teaching examples are not production benchmarks; confirm APIs and model support against the linked version.
