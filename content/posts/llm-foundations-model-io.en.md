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

## A complete explanation from first principles

### Identify the task before interpreting an output

“Model output” has no universal meaning. A generative model returns vocabulary logits for next-token prediction. An embedding model returns representations for similarity or retrieval. A classifier returns class scores. Treating the first embedding dimension as a sentiment probability, or assuming an arbitrary hidden state is a retrieval-trained sentence vector, requires evidence that is usually absent.

This chapter includes complementary experiments. The browser MiniLM inspector loads real pretrained weights and displays tokens, masks, sentence vectors and cosine similarity. The full PyTorch notebook inspects DistilGPT2, a generative model. Their vocabularies, hidden dimensions and attention patterns are not interchangeable.

### Trace four different tensors

`input_ids` commonly has integer shape `[B,T]`; an attention mask identifies valid positions. Embedding lookup produces floating-point `[B,T,d]` representations. Contextual layers commonly preserve that shape. A language-model head then produces `[B,T,V]`, where V is vocabulary size.

For `B=1,T=6,d=768,V=50257`, a hidden state contains 4,608 elements, while logits contain 301,542. These are not two names for the same output. Returned attention may additionally have a head axis, such as `[1,12,6,6]`. Requesting every layer's attention can consume substantial memory, and optimized kernels may not materialize those matrices directly.

### Reduce token vectors to a sentence vector deliberately

Retrieval often requires one vector per sentence. Masked mean pooling sums valid token representations and divides by their count. Including padding can make a representation depend on unrelated batch-padding choices.

L2 normalization then makes vector length one, allowing a dot product to equal cosine similarity. Vector `[3,4]` has length five and normalizes to `[0.6,0.8]`. Normalized features are not probabilities: components may be negative and need not sum to one.

```python
import math
u, v = [3,4], [4,3]
def norm(x): return math.sqrt(sum(a*a for a in x))
cosine = sum(a*b for a,b in zip(u,v))/(norm(u)*norm(v))
print(cosine)  # 0.96
```

### Similarity is not factual correctness

Nearby embeddings indicate closeness according to the model's learned similarity structure. Negation, entity replacement and numerical changes can preserve topic similarity while reversing the useful answer. “Refunds are allowed” and “refunds are not allowed” discuss the same subject but imply opposite actions. Cosine similarity is neither a calibrated business probability nor a factual verifier.

Try related English sentences in the actual MiniLM inspector, then change only a date, amount or negation. Observe whether those changes are large enough for your retrieval task. This checkpoint primarily targets English sentence embeddings; its Chinese behavior does not establish the capabilities of multilingual embedding models in general.

### Ensure the displayed input is the actual input

A tokenizer length limit must apply to the tensor passed into the forward call, not merely to a separate display tokenization. Otherwise an interface could show 128 tokens while the model reads something else. The inspector uses the same encoded input for display and forward execution.

Also record pooling rules, treatment of special tokens and the selected layer. Hold those choices constant when comparing representations. Evaluation mode and fixed inputs help distinguish model differences from implementation differences.

### Treat failed execution as a useful observation

The first browser inference downloads runtime code and model weights; subsequent forward computation occurs locally. Network or memory failures are reported rather than replaced with canned vectors. The standard-Python notebook provides inspectable small-model training. The full PyTorch tab preserves actual CPU outputs and an execution link for a compatible environment; the browser kernel cannot install an ordinary CUDA PyTorch build.

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
