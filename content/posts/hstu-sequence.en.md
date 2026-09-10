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
