---
title: "Reconstruction: what does a model learn from hidden patches?"
date: "2026-09-10"
updated: "2026-09-10"
category: "Multimodal Understanding"
language: "en"
tags: ["multimodal", "Hands-on", "Model Learning"]
pairKey: "multimodal-reconstruction"
slug: "multimodal-reconstruction-en"
excerpt: "Distinguish pixel reconstruction from semantic prediction; inspect MAE visible-only encoding, lightweight decoding and masked loss."
series: "multimodal"
seriesOrder: 3
draft: false
---

![Reconstruction: what does a model learn from hidden patches?](/learning/multimodal-reconstruction-en.svg)

## Hide inputs to create supervision

Masked autoencoding removes patches and predicts missing content from visible regions. The original image supplies targets without manual category labels. MAE encodes only visible patches, then inserts mask tokens and positions in a decoder to restore sequence order. Downstream tasks commonly reuse the encoder, not the reconstruction decoder as a recognition head.

For 196 patches and 75% masking, 49 remain visible. CLS and implementation details may add another token. Shortening the encoder sequence does not reduce total model compute by exactly 75%, because decoding and projections remain.

## Which positions contribute to loss?

A simplified masked MSE is $\sum_i m_i\|\hat p_i-p_i\|^2 / \sum_i m_i$, with squared error averaged within each patch and m=1 meaning hidden. This differs from the common attention-mask convention where one means valid. Supervising hidden patches avoids rewarding only direct copying.

```python
import torch
pred = torch.zeros(1, 4, 3)
target = torch.ones_like(pred)
mask = torch.tensor([[1.,0.,1.,0.]])
per_patch = ((pred-target)**2).mean(-1)
loss = (per_patch*mask).sum()/mask.sum()
assert loss.item() == 1.0
```

Normalizing patch targets changes the loss and reconstruction display. Inspect settings such as `norm_pix_loss`; arbitrary logits are not automatically 0–255 pixels.

## Attractive reconstruction is not stronger understanding

Pixel objectives reward texture and low-level detail; semantic tasks may require invariance to lighting or background. Latent-prediction approaches change the target space but must address issues such as representation collapse. Compare reconstruction error and transfer performance separately. A visually pleasing completion does not establish better retrieval or classification.

The on-page experiment uses synthetic grayscale patches and a constant predictor. It is untrained. The notebook loads actual `facebook/vit-mae-base` weights, fixes the random mask and displays the original, masked and reconstructed image together with model loss.

## Check your understanding

**Question:** Does zero loss with zero mask ratio imply perfection?

<details><summary>Answer</summary>No hidden positions are supervised and the denominator is zero. Training should reject the setting or define empty loss explicitly. The calculator's zero is only a display convention.</details>

## Actual output from this run

![Actual vit-mae-base reconstruction of a synthetic color grid. This is neither a hand-filled image nor a natural-image quality benchmark.](/learning/multimodal-reconstruction-actual.png)

Actual vit-mae-base reconstruction of a synthetic color grid. This is neither a hand-filled image nor a natural-image quality benchmark.

## Primary sources and further reading

- [Masked Autoencoders](https://arxiv.org/abs/2111.06377)
- [ViTMAE implementation](https://huggingface.co/docs/transformers/model_doc/vit_mae)
- [I-JEPA](https://arxiv.org/abs/2301.08243)

Sources checked on 2026-09-10. Teaching examples are not production benchmarks; confirm APIs and model support against the linked version.
