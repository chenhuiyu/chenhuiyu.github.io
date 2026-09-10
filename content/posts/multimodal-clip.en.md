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

## A complete explanation from first principles

### A learned space makes cross-modal comparison possible

Pixels and text tokens cannot be compared meaningfully by directly multiplying their raw arrays. CLIP uses separate image and text encoders and trains their vectors into a shared comparison space. Shared geometry does not imply identical encoder architectures.

A cat photograph should match “a photo of a cat” better than “a red bicycle.” Paired image-text data supplies relative supervision without requiring a fixed category label for every image.

### Treat the batch as a matching problem

For N paired examples, image and text matrices both have shape `[N,d]`. After normalization, their matrix product gives an `[N,N]` similarity table. Row i should select text i; the reverse direction should select image i.

For two pairs, scores `[[8,2],[3,7]]` put correct matches on the diagonal. Compute cross-entropy over each row, then over rows of the transposed matrix, and average. The browser notebook calculates both directions. One row's loss is not the entire training objective.

### Not every off-diagonal pair is truly wrong

A batch can contain multiple cat images and nearly interchangeable captions. Treating all off-diagonal pairs as negatives is convenient but creates semantically plausible false negatives. Duplication and generic captions affect the supervision signal, so larger batches should not substitute for data inspection.

Temperature controls score sharpness. Extremely sharp distributions can concentrate gradients on difficult or mislabeled examples; overly smooth ones weaken discrimination. Data, encoders and negative composition matter alongside temperature.

### Zero-shot classification is candidate-text comparison

Encode a prompt for each category, compare an image against those vectors and choose the highest score. New category names can participate without training a dedicated classification head, but wording affects results.

If candidates are only cat and dog, a car image may still be assigned one of them. Softmax scores are relative to the supplied candidate set, not automatically calibrated open-world confidence. Adding a category changes the normalized probabilities of existing categories.

### Global similarity does not solve every understanding task

“The red cup is left of the blue plate” and a phrase with swapped attributes share many words while expressing different relationships. Global embeddings can capture topic similarity without reliably handling counting, spatial relations, negation or attribute binding.

Construct minimal contrasts that change only left/right, quantity or negation. They reveal failure modes more clearly than a small random gallery. Product retrieval additionally needs slices for new items, fine-grained categories and variants of the same product.

The browser edition uses hand-built two-dimensional vectors to expose the loss calculation; it does not present them as pretrained CLIP outputs. The full PyTorch tab contains actual image/text encoding and saved execution results. Verify preprocessing and prompt templates before interpreting scores, and do not equate low loss on a synthetic example with real retrieval quality.

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
