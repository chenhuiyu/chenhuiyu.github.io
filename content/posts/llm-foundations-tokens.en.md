---
title: "How does text become a tensor?"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Foundations"
language: "en"
tags: ["llm-foundations", "Hands-on", "Model Learning"]
pairKey: "llm-foundations-tokens"
slug: "llm-foundations-tokens-en"
excerpt: "Understand vocabulary IDs, embedding lookup and padding through the four dimensions B, T, D and V."
series: "llm-foundations"
seriesOrder: 1
draft: false
---

![How does text become a tensor?](/learning/llm-foundations-tokens-en.svg)

## Start with one sentence

A language model does not directly receive “The cat is sleeping.” A tokenizer maps the string into integer IDs, and an embedding table maps each ID to a vector. A token can be a word, subword, byte fragment or special symbol. Character counts and word counts are not interchangeable with token counts. Tokenization is discrete preprocessing; embeddings are learned parameters.

Imagine an indexed dictionary, but do not attach numerical meaning to the indices. IDs 120 and 121 need not be semantically related. Training changes their vectors, not the integers. A tokenizer must match the model's vocabulary and embedding table.

## Follow the shapes

Let batch size $B=2$, maximum length $T=5$, hidden dimension $D=8$ and vocabulary size $V=100$. IDs have integer shape `[2,5]`, the table is `[100,8]`, and lookup returns floating-point `[2,5,8]`. Later layers mix context into those vectors. An initial lookup embedding and a final contextual representation are different objects.

```python
import torch
embedding = torch.nn.Embedding(100, 8)
ids = torch.tensor([[4, 12, 9, 0, 0], [6, 7, 8, 9, 10]])
mask = ids.ne(0)  # Only this example defines zero as padding.
x = embedding(ids)
assert x.shape == (2, 5, 8)
print(mask.sum(dim=1))  # tensor([3, 5])
```

## Why padding contaminates averages

Padding turns unequal sequences into a rectangular tensor, but padding positions are not text. If three valid scalar representations are `[1,2,3]` and two padding values are zero, a naive mean is 1.2; the masked mean is 2. Real padding embeddings need not even be zero. An attention mask identifies valid context positions. A loss mask selects supervised positions. They solve different problems.

Retrieval often pools `[B,T,D]` into `[B,D]`. Generation instead projects to `[B,T,V]`. When inspecting an output, first identify whether it contains token states, sentence vectors or vocabulary logits. A large final dimension alone tells you little.

## Experiment and check

In the real-model I/O chapter, change capitalization, punctuation and language while recording valid token counts. Inspect padding before interpreting similarities. Successful tokenization does not establish semantic understanding, especially outside the training distribution.

**Question:** If every token ID increases by one while weights remain fixed, have we merely renamed the vocabulary?

<details><summary>Answer</summary>No. The model looks up different embedding rows. Preserving behavior would require a consistent permutation of the tokenizer mapping and every relevant vocabulary-indexed parameter.</details>

## Primary sources and further reading

- [Transformers: GPT-2 model and outputs](https://huggingface.co/docs/transformers/model_doc/gpt2)

Sources checked on 2026-09-10. Teaching examples are not production benchmarks; confirm APIs and model support against the linked version.
