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
