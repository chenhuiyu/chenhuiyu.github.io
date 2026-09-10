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

## A complete explanation from first principles

### Separate pieces of text, integer IDs and vectors

A computer stores encoded bytes, while a neural network computes with numbers. A tokenizer establishes the interface between them. It segments the input into vocabulary-supported pieces, maps those pieces to integer IDs, and passes the IDs to an embedding lookup. These are three different objects. ID 100 is not more important than ID 20; the numbers are addresses in a table, not semantic measurements.

A vector is a list of numbers. A matrix is a table of numbers. A tensor allows additional axes. When you encounter a shape such as `[2,5,8]`, label its axes before doing arithmetic: it might mean two examples, five padded positions per example, and eight features per position. Shapes describe organization, not the values stored inside.

### Work through an embedding lookup

Consider a vocabulary `['<pad>', 'cat', 'sits', 'mat']` and an embedding table with four rows and three columns. Under this deliberately simple tokenizer, “cat sits” becomes `[1,2]`. Selecting rows 1 and 2 produces a `[2,3]` matrix. Adding the batch axis for one example gives `[1,2,3]`.

```python
vocab = {'<pad>': 0, 'cat': 1, 'sits': 2, 'mat': 3}
E = [[0,0,0], [0.2,0.5,-0.1], [0.8,-0.2,0.3], [-0.1,0.4,0.9]]
ids = [vocab[word] for word in ['cat','sits']]
x = [E[i] for i in ids]
print(ids, x)
```

This uses only standard Python and can run in the embedded notebook. Real tables are larger, but the indexing relationship is identical. Embeddings are learned parameters, not hand-written definitions. An ID initially selects the same row wherever it occurs; later contextual layers can give that token different representations in different sentences.

### Why tokenizers use pieces smaller than words

Whitespace splitting handles some English examples but fails for languages without spaces, spelling variations, source code and unfamiliar words. Subword tokenization trades vocabulary size against sequence length. Common fragments receive their own IDs; uncommon strings can be assembled from smaller pieces. BPE builds a vocabulary by repeatedly merging smaller units according to corpus statistics. Real implementations also have preprocessing, byte handling and special-token rules.

A character is therefore not guaranteed to equal a token. Token IDs from one model cannot generally be fed to another model. The tokenizer and the weight checkpoint form an interface contract. A tokenizer may encode one language less compactly than another, affecting context budgets and computation, without that observation alone establishing the model's language capability.

### Padding and truncation can hide semantic mistakes

A batch containing sequences of length three and five is often stored as `[2,5]`. The two added positions in the shorter example are storage padding. An attention mask excludes inappropriate positions from reading; a loss mask excludes positions from supervision. Those jobs are different. A valid attention mask does not automatically guarantee that padding has been removed from the loss.

Truncation discards positions beyond a configured limit. A program can produce perfectly valid tensors after silently removing the only sentence containing the answer. During debugging, inspect the input string, IDs, decoded tokens, mask and retained length together. Successful execution is weaker evidence than semantic correctness.

### Connect the representation to the experiment

The first browser-notebook cell intentionally builds a character vocabulary so every mapping remains visible. Change the text and rerun it. If the vocabulary changes, downstream parameter tables must be rebuilt as well. This is a teaching tokenizer, not a replacement for the pretrained tokenizer used in the model I/O chapter.

Finally, a vocabulary of 10,000 entries with a hidden dimension of 256 has 2,560,000 embedding parameters. That count does not depend on the current prompt length. Longer prompts create larger intermediate tensors and more computation; they do not dynamically add rows to the learned vocabulary table.

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
