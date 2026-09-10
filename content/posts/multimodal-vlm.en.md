---
title: "Vision-language models: from visual features to answers"
date: "2026-09-10"
updated: "2026-09-10"
category: "Multimodal Understanding"
language: "en"
tags: ["multimodal", "Hands-on", "Model Learning"]
pairKey: "multimodal-vlm"
slug: "multimodal-vlm-en"
excerpt: "Understand the vision encoder/projector/decoder interface and distinguish alignment, instruction tuning and grounding."
series: "multimodal"
seriesOrder: 4
draft: false
---

![Vision-language models: from visual features to answers](/learning/multimodal-vlm-en.svg)

## Different models, different outputs

CLIP usually produces matching scores. A generative VLM produces text conditioned on images and a question. One common architecture encodes `[B,N,Dv]`, projects features into language hidden dimension Dl and inserts them according to the model's input protocol. The projector addresses dimensions and representation compatibility; it does not simply translate an image into a sentence.

Other models use cross-attention, resamplers or token merging. Special tokens, image placement and positions must match the processor and chat template.

## Training involves distinct decisions

An alignment stage may freeze backbones while training a bridge. Visual instruction tuning supervises answers to image/question pairs. Which modules update depends on the recipe. Language priors can conceal weak vision: if nearly every banana question expects yellow, the model may answer without inspecting the image. Change the image while fixing the question to test dependence on evidence.

```python
import torch
visual = torch.randn(1, 196, 768)
projector = torch.nn.Linear(768, 1024)
visual_tokens = projector(visual)
text_tokens = torch.randn(1, 12, 1024)
combined = torch.cat([visual_tokens, text_tokens], dim=1)
assert combined.shape == (1, 208, 1024)
```

This illustrates the shape interface, not a complete VLM. It omits processing, modality markers, positions/masks and learned alignment.

## Grounding requires spatial evidence

“There is a cat” differs from localizing the cat relative to the red sofa. Coordinates may be pixels, normalized values or discrete bins; resizing/cropping requires mapping back to the original image. OCR needs adequate resolution, and fine attributes need local evidence. Fluent language cannot replace those checks.

When reading reports such as Qwen2.5-VL, inspect dynamic resolution, visual-token budgets and temporal representations in relation to tasks. Advertised context capacity does not imply efficient operation on every device.

## Check your understanding

**Question:** Accuracy is unchanged when images are hidden. Does that demonstrate a robust vision encoder?

<details><summary>Answer</summary>It suggests possible question or dataset shortcuts. Compare shuffled images, blank images, question-only inputs and counterfactual images to test whether answers depend on visual evidence.</details>

## Primary sources and further reading

- [LLaVA / Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
- [Qwen2.5-VL report](https://arxiv.org/abs/2502.13923)

Sources checked on 2026-09-10. Teaching examples are not production benchmarks; confirm APIs and model support against the linked version.
