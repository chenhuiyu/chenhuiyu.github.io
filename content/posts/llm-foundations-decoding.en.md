---
title: "From logits to text: sampling, KV cache and stopping"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Foundations"
language: "en"
tags: ["llm-foundations", "Hands-on", "Model Learning"]
pairKey: "llm-foundations-decoding"
slug: "llm-foundations-decoding-en"
excerpt: "Compute softmax, temperature and top-p; distinguish decoding policy from model distribution and inspect cache reuse."
series: "llm-foundations"
seriesOrder: 5
draft: false
---

![From logits to text: sampling, KV cache and stopping](/learning/llm-foundations-decoding-en.svg)

## A complete explanation from first principles

### A forward pass is not yet a complete answer

A language-model forward pass produces vocabulary scores at input positions. Generation usually selects the final valid position's logits, chooses one next token, appends it to the context and repeats. A `[B,T,V]` tensor does not mean the model has already generated T future tokens simultaneously.

Logits can be negative and do not sum to one. Softmax converts them into probabilities. The checkpoint, decoding policy and stopping rules jointly determine visible output. Reproducing a result requires more than naming the model weights.

### Calculate what temperature does

For logits `[2,1]`, temperature one gives approximately `[0.731,0.269]`. Temperature 0.5 applies softmax to `[4,2]`, yielding approximately `[0.881,0.119]`. Temperature two gives approximately `[0.622,0.378]`. Lower temperature sharpens preferences; it does not inject factual knowledge.

Greedy decoding selects the maximum. Top-k retains the k highest-scoring candidates, while top-p retains a set determined by cumulative probability. The remaining distribution is renormalized. The order of temperature and filtering operations matters, so inspect the library's behavior. Directly dividing by zero is invalid; APIs generally express greedy decoding through a separate setting.

### KV caching reuses computation rather than answers

In a causal model, representations of previous tokens should not depend on tokens generated later. Their keys and values can therefore be reused. Prefill processes the prompt; subsequent decode steps compute representations for newly added tokens and read the accumulated KV history.

Caches are stored across layers. With 32 layers, eight KV heads, head dimension 128 and two bytes per element, each token requires `2*32*8*128*2=131072` bytes, or 128 KiB. A 4,096-token sequence requires approximately 512 MiB; eight such sequences require roughly 4 GiB, excluding weights, temporary activations and allocator overhead.

GQA and MQA reduce KV head count. They do not simply shorten generated text. Caching avoids redundant historical computation, but the new query still reads historical keys and values, so decode traffic can increase with context length.

### Test cache correctness with logits

Run the same prefix through full recomputation and incremental cached execution. Compare next-token logits with a numerical tolerance. Large discrepancies suggest incorrect positions, masks, cache lengths or accidentally feeding the last token twice.

Use evaluation mode to remove dropout from the comparison. Different attention kernels may introduce small floating-point differences. Comparing only final generated text is weak: tiny score changes can alter a sampled trajectory, while matching text does not prove that every intermediate operation is correct.

### Stopping rules affect the meaning of a response

EOS is a vocabulary symbol; a maximum-new-token limit is a resource budget. They serve different purposes. Hitting a length cap can leave JSON, code or citations incomplete. String-based stopping also raises token-boundary questions and whether the stop string should remain visible.

The browser notebook samples from an actually trained bigram model. Hold the random seed fixed and compare temperatures 0.3, one and two. Record repetition and coherence before drawing conclusions. This small model lacks long-range understanding, so its behavior illustrates sampling rather than establishing how every LLM behaves. The full PyTorch notebook separately checks real autoregressive-model caches and logits.

## The model scores; the decoder selects

The final position's logits are unnormalized vocabulary scores. Greedy decoding chooses the maximum; sampling draws from probabilities. Temperature uses $p_i\propto\exp(z_i/T)$, with lower positive T usually concentrating mass. Greedy should be a separate mode rather than literal division by zero. Top-k retains k candidates. Top-p sorts probabilities and retains the smallest prefix reaching a cumulative threshold, then renormalizes.

For `[0.6,0.25,0.15]` at top-p=0.8, retain the first two entries, whose mass is 0.85. Their new probabilities are approximately `[0.706,0.294]`. The retained set changes every step. Applying temperature before or after truncation can also change results.

## Prefill and decode do different work

Prefill processes the known prompt and builds layer-wise K/V tensors. Decode produces a new position's Q/K/V while reading existing cached keys and values. A typical per-sequence, per-layer cache has shape `[Hkv,T,Dh]`; actual batch layout varies. Caching avoids recomputing old projections, but a new query still accesses context. Standard dense attention's per-step reads grow with T.

A KV cache is neither an answer cache nor a store of every hidden state. Changing earlier prompt tokens, positions or model weights can invalidate it. Paging and prefix sharing are system-level choices.

## A falsifiable cache experiment

The notebook compares full-prompt logits with a two-step computation: obtain `past_key_values` for the prefix, then feed only the final token. Last-position logits should agree within numerical tolerance. Padded batches require correct masks, position IDs or cache positions; a single-sequence example cannot be generalized blindly.

Stopping is part of the decoder: EOS, maximum new tokens, stop strings and tool-call protocols are different mechanisms. Record whether generation hit a length limit before diagnosing an incomplete answer as a modeling failure.

## Check your understanding

**Question:** If changing temperature produces a better answer, did the model learn new knowledge?

<details><summary>Answer</summary>No weights changed. Only selection from the existing distribution changed. Use repeated samples, a fixed evaluation budget and correctness metrics to establish whether the improvement is reliable.</details>

## Primary sources and further reading

- [Transformers: GPT-2 model and outputs](https://huggingface.co/docs/transformers/model_doc/gpt2)
- [The generation API](https://huggingface.co/docs/transformers/main_classes/text_generation)

Sources checked on 2026-09-10. Teaching examples are not production benchmarks; confirm APIs and model support against the linked version.
