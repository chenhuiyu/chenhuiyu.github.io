---
title: "From pretraining to alignment: what is optimized?"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Foundations"
language: "en"
tags: ["llm-foundations", "Hands-on", "Model Learning"]
pairKey: "llm-foundations-training"
slug: "llm-foundations-training-en"
excerpt: "Separate next-token loss, SFT, LoRA and preference optimization; inspect shifted labels and supervision masks."
series: "llm-foundations"
seriesOrder: 4
draft: false
---

![From pretraining to alignment: what is optimized?](/learning/llm-foundations-training-en.svg)

## A complete explanation from first principles

### Turn a prediction into a learning signal

A model begins as adjustable numerical parameters. Given a context, it predicts probabilities for the next token. The actual next token supplies a label. If the correct class receives probability 0.7, its negative log-likelihood is about 0.357; at probability 0.1, the loss is about 2.303. Assigning less probability to the observed target receives a larger penalty.

Loss is an optimization objective, not a general intelligence score. Better prediction can support useful capabilities without guaranteeing factual correctness or agreement with every user's preferences.

### Derive the gradient you will use repeatedly

Let z be logits, p their softmax probabilities, and y a one-hot target. Cross-entropy has logit gradient `p-y`. If the correct class currently receives probability 0.7, its gradient is -0.3, so gradient descent increases its logit. Incorrect classes receive positive gradients proportional to their predicted probabilities. The chain rule carries this signal back into earlier parameters.

The browser notebook explicitly implements this expression while training a bigram language model. Its loss curve comes from actual parameter updates, not a saved animation. The model only learns neighboring-character relationships, but that small setting makes learning rate, epochs and objective behavior easy to inspect.

### Shift the supervision correctly

A sequence such as “I love cats” supplies predictions for “love cats EOS.” Multiple positions can be supervised in one forward pass, while a causal mask prevents future information from leaking into each prediction. Some model APIs shift labels internally. Others expect the caller to prepare shifted tensors. Shifting twice silently changes the task.

During SFT, system and user text often serve as conditions while assistant tokens receive the primary loss. Training on user tokens as targets changes the objective. Ignoring a label position removes its loss contribution; it does not automatically prevent other positions from attending to that input.

### Batch size is also a weighting decision

A batch averages information from several examples before updating parameters. Gradient accumulation splits a larger effective batch into memory-sized microbatches and delays the optimizer step. If each microbatch first averages over its own token count, then averaging those losses equally gives different token weights when lengths differ substantially. Decide whether the desired objective averages over examples or valid tokens, and implement that denominator consistently.

Large learning rates can cause oscillation or divergence; small ones can make progress slow. Falling training loss is insufficient evidence of improvement. A rising validation loss may indicate overfitting, while data leakage can make both curves look deceptively good. The split protocol belongs to the training method.

### Separate training stages from parameter-update methods

Pretraining and SFT describe data and objective stages. LoRA describes how selected parameters are updated: a low-rank increment modifies a frozen weight matrix. It can therefore be used during SFT. For a `[4096,4096]` matrix, rank eight requires 65,536 low-rank parameters rather than 16,777,216 full-matrix parameters. Fewer trainable parameters do not eliminate the base model's forward computation or weight storage.

DPO uses preferred and rejected responses for the same prompt, comparing policy and reference-model log probabilities. It is not simply ordinary SFT on the preferred response. Nor does preference optimization guarantee coverage of every domain. Comparisons must control data, compute budget and evaluation rules before attributing gains to the algorithm.

### Make an experiment reproducible

Record the tokenizer, configuration, seed, dataset version, split, optimizer and learning-rate schedule. Evaluate task quality, language and length slices, formatting failures and concrete error cases alongside average loss. Being able to explain exactly how labels produced a reported loss is more valuable than merely recognizing the names of training methods.

## Parallel training, sequential generation

The autoregressive factorization is $p(x)=\prod_t p(x_t\mid x_{<t})$. Training already has the full sequence. A causal mask prevents future access, allowing many next-token losses in one forward pass. At generation time future tokens do not yet exist, so decoding proceeds sequentially.

Inputs `[A,B,C]` correspond to targets `[B,C,D]`. Cross-entropy is the negative log probability of the target. Lower loss measures fit to that distribution; it does not directly establish factuality, instruction following or reliable reasoning.

## SFT changes supervision; LoRA changes parameterization

Supervised fine-tuning commonly trains on instruction-response sequences, optionally computing loss only on assistant tokens. Explicitly document whether system prompts, user text and padding are supervised. LoRA uses a low-rank update $\Delta W=BA$ for selected linear layers. It is a parameterization method rather than a different task objective, so SFT and LoRA can be combined.

For `[d_out,d_in]` weights, rank r requires $r(d_{in}+d_{out})$ update parameters. Fewer trainable parameters reduce optimizer state, but do not eliminate base weights or activation memory.

```python
import torch
import torch.nn.functional as F
logits = torch.randn(1, 3, 5, requires_grad=True)
labels = torch.tensor([[1, 2, -100]])
loss = F.cross_entropy(logits.reshape(-1, 5), labels.reshape(-1),
                       ignore_index=-100)
loss.backward()
assert torch.all(logits.grad[0, 2] == 0)
```

Here logits and labels are already aligned. A Hugging Face causal LM's built-in loss typically shifts ordinary labels internally; do not shift twice.

## What preferences add

DPO compares chosen and rejected answers for the same prompt, adjusting log-probability ratios relative to a fixed reference policy. It depends on preference quality and modeling assumptions. RLHF and RLVR further distinguish learned reward models from programmatically verifiable rewards. Improving reward may exploit weaknesses in the evaluator; measure independent outcomes as well.

## Check and diagnose

**Question:** Training loss keeps falling while validation worsens. Should you simply train longer?

<details><summary>Answer</summary>First inspect overfitting, duplication, shifted labels, prompt-template differences and leakage. More steps may amplify the issue. Overfitting a tiny batch checks implementation; measuring a held-out set checks generalization. These are separate experiments.</details>

## Primary sources and further reading

- [Transformers: GPT-2 model and outputs](https://huggingface.co/docs/transformers/model_doc/gpt2)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [DPO](https://arxiv.org/abs/2305.18290)

Sources checked on 2026-09-10. Teaching examples are not production benchmarks; confirm APIs and model support against the linked version.
