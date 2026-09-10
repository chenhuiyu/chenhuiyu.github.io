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
