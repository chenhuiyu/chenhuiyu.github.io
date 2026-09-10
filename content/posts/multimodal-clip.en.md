---
title: "CLIP: bring images and text into a shared space"
date: "2026-09-10"
updated: "2026-09-10"
category: "Multimodal Understanding"
language: "en"
tags: ["multimodal", "Hands-on", "Model Learning"]
pairKey: "multimodal-clip"
slug: "multimodal-clip-en"
excerpt: "Construct a B×B similarity matrix and understand contrastive training, negatives, temperature and zero-shot classification."
series: "multimodal"
seriesOrder: 2
draft: false
---

![CLIP: bring images and text into a shared space](/learning/multimodal-clip-en.svg)

## Two encoders, one matching task

Image and text encoders produce projected, normalized vectors. With B paired examples, $S_{ij}=u_i^\top v_j/\tau$ has shape `[B,B]`. Diagonal entries are paired examples; other entries typically act as in-batch negatives. Image-to-text and text-to-image cross-entropies form a symmetric objective.

Off-diagonal does not necessarily mean semantically wrong. Two cat images may both match “a cat.” False negatives and repeated captions affect learning, so batch construction and cleaning are part of the method.

## Temperature changes competition

Small temperature magnifies similarity differences, sharpening distributions and altering gradients. Multiplying all similarities by a large value can encourage overconfidence. A learned training logit scale is not the same operation as manually changing a decision threshold.

```python
import torch
import torch.nn.functional as F
image = F.normalize(torch.randn(4, 8), dim=-1)
text = F.normalize(torch.randn(4, 8), dim=-1)
s = image @ text.T / .07
y = torch.arange(4)
loss = (F.cross_entropy(s,y)+F.cross_entropy(s.T,y))/2
print(loss.item())
```

Random vectors check shapes and objective implementation. The notebook loads real CLIP weights for an image and candidate descriptions you provide.

## Zero-shot classification depends on the candidate set

Encode category descriptions, compare them with the image and normalize scores. These probabilities are relative to the supplied labels. An image outside every category still gets a maximum. Adding a similar label changes the denominator. Wording, language and category granularity affect ranking.

Content understanding also includes relationships, counting, OCR, temporal order and fine attributes. Global similarity does not establish those abilities. Try counterfactual descriptions to test whether the model distinguishes relationships rather than merely matching object words.

## Check your understanding

**Question:** One candidate label receives softmax probability one. Is the model certain?

<details><summary>Answer</summary>No. A one-element softmax is always one. Unknown-class handling requires suitable raw-score interpretation, rejection methods and held-out evaluation.</details>

## Primary sources and further reading

- [CLIP](https://arxiv.org/abs/2103.00020)
- [Transformers CLIP](https://huggingface.co/docs/transformers/model_doc/clip)

Sources checked on 2026-09-10. Teaching examples are not production benchmarks; confirm APIs and model support against the linked version.
