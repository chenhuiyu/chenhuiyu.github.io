---
title: "A real-model lab: tokens, hidden states, attention and embeddings"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Foundations"
language: "en"
tags: ["llm-foundations", "Hands-on", "Model Learning"]
pairKey: "llm-foundations-model-io"
slug: "llm-foundations-model-io-en"
excerpt: "Load MiniLM in the browser and GPT-2 in a notebook; inspect actual tensors to distinguish encoders, decoders and pooling."
series: "llm-foundations"
seriesOrder: 6
draft: false
---

![A real-model lab: tokens, hidden states, attention and embeddings](/learning/llm-foundations-model-io-en.svg)

## Decide what to observe

“Load a model” hides three stages: tokenization, the forward pass and task-specific postprocessing. The browser uses `Xenova/all-MiniLM-L6-v2`, a lightweight encoder for semantic similarity. It is not an autoregressive chat model.

The button downloads real ONNX weights and runs CPU/WASM inference locally. A failure produces an error, never a canned similarity score. The first run needs runtime and weight downloads. An English-oriented model does not gain equivalent Chinese retrieval quality because the interface is bilingual.

## From token states to a sentence vector

The model produces token representations, followed by masked mean pooling and L2 normalization. For valid-token indicators m, $e=\sum_t m_th_t/\sum_t m_t$, then divide by $\|e\|_2$. The dot product of normalized vectors is cosine similarity, in [-1,1], not a calibrated confidence probability.

Inspect real token IDs, padding masks, output shape `[2,384]` and the first 32 dimensions. An individual embedding coordinate rarely has a stable nameable meaning. The bars visualize values, not attention.

## Go deeper in the notebook

The companion loads actual `distilgpt2` weights and inspects `[B,T,V]` logits, layer-wise hidden states and `[B,H,T,T]` attention. It also compares cached and full forward passes. Eager attention is selected to expose weights; optimized attention backends may not return the same observables.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained('distilgpt2')
model = AutoModelForCausalLM.from_pretrained(
    'distilgpt2', attn_implementation='eager').eval()
inputs = tok('The cat is', return_tensors='pt')
# The complete notebook adds no_grad, plots and cache equivalence checks.
```

## Three useful counterexamples

Compare a sentence with a paraphrase, then add a negation, then try domain terminology or Chinese. High lexical overlap can still produce high similarity despite opposite logical meaning. Make an input longer than 128 tokens and inspect whether truncation removes critical information. Save inputs, effective length and model version with results.

**Question:** Does cosine similarity 0.9 mean a 90% probability of mutual entailment?

<details><summary>Answer</summary>No. It measures geometric proximity in the representation space. Entailment needs an appropriate task and evaluation set; an uncalibrated similarity score does not supply that probability.</details>

## Actual output from this run

![Actual DistilGPT2 attention, layer 0 head 0. The zero upper triangle reflects causal masking.](/learning/llm-foundations-model-io-actual.png)

Actual DistilGPT2 attention, layer 0 head 0. The zero upper triangle reflects causal masking.

## Primary sources and further reading

- [MiniLM ONNX model](https://huggingface.co/Xenova/all-MiniLM-L6-v2)
- [Transformers.js 3.8.1](https://huggingface.co/docs/transformers.js/v3.8.1/pipelines)
- [Transformers: GPT-2 model and outputs](https://huggingface.co/docs/transformers/model_doc/gpt2)

Sources checked on 2026-09-10. Teaching examples are not production benchmarks; confirm APIs and model support against the linked version.
