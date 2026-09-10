---
title: "How images become tokens: patches, positions and resolution"
date: "2026-09-10"
updated: "2026-09-10"
category: "Multimodal Understanding"
language: "en"
tags: ["multimodal", "Hands-on", "Model Learning"]
pairKey: "multimodal-patches"
slug: "multimodal-patches-en"
excerpt: "Trace RGB tensors into ViT patch embeddings, count visual tokens and inspect information lost through resizing and cropping."
series: "multimodal"
seriesOrder: 1
draft: false
---

![How images become tokens: patches, positions and resolution](/learning/multimodal-patches-en.svg)

## Images are not pixel sentences

A vision encoder often divides an image into patches, flattens each and projects it into D features. For `[B,3,H,W]` and patch width P, divisible dimensions give $N=(H/P)(W/P)$ tokens with $3P^2$ raw features each. This is a learned linear transformation, not a discrete vocabulary lookup. Some models add CLS; others pool or compress differently.

A 224×224 image with 16×16 patches gives 196 patch tokens, or 197 including CLS. At 448×448, there are 784 patches and approximately sixteen times as many dense-attention pairs, not merely twice the cost.

## Runnable patchification

```python
import torch
image = torch.arange(3*32*32).reshape(1,3,32,32).float()
p = 8
patches = image.unfold(2,p,p).unfold(3,p,p)
patches = patches.permute(0,2,3,1,4,5).reshape(1,16,3*p*p)
projection = torch.nn.Linear(3*p*p, 64)
print(projection(patches).shape)  # [1,16,64]
```

Permutation order determines pixel layout and must match unpatchification. Correct shapes alone do not establish correct channels or positions. The notebook verifies round-trip patchification against the original tensor.

## Preprocessing belongs to the model

Resizing, cropping, RGB conversion and normalization affect outputs. Passing already-rescaled float pixels into a processor that rescales by default can divide twice. Cropping a long image to a square may remove edge text before the encoder sees it. Inspect the processed image before blaming the generator.

Dynamic-resolution models can produce different visual-token counts for different aspect ratios; video introduces time as well. The fixed-ViT 196-token example does not apply universally.

## Check your understanding

**Question:** Can patch width change from 16 to 8 while retaining the original patch-projection weights unchanged?

<details><summary>Answer</summary>Usually not. Flattened patch size changes from 768 to 192, so weight shapes differ; positional representations and training distributions also change. Adaptation requires a defined method, not just a configuration edit.</details>

## Primary sources and further reading

- [Vision Transformer](https://arxiv.org/abs/2010.11929)
- [Image processors](https://huggingface.co/docs/transformers/main_classes/image_processor)

Sources checked on 2026-09-10. Teaching examples are not production benchmarks; confirm APIs and model support against the linked version.
