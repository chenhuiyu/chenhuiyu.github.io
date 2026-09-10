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

## A complete explanation from first principles

### A vision encoder needs an interface to a language model

A language model expects token representations such as `[B,T,d_text]`, while a vision encoder may produce `[B,N,d_vision]` patch features. Sequence meanings and feature dimensions differ. A projector maps visual features into a compatible representation, or cross-attention provides another interaction mechanism.

Matching dimensions establishes interface compatibility, not semantic alignment. Training still needs to teach the model how visual information should influence an answer.

### Trace one concrete visual sequence

Suppose the vision encoder returns 196 patches of width 768, and a projector maps them to width 4096. An early-fusion example with thirty text tokens can contain at least 226 positions, plus possible separators. This is one layout, not a universal VLM rule.

Other models use resampling, cross-attention, multiple crops or dynamic resolution. An image does not always correspond to 196 tokens. Inspect the actual processed count before budgeting context length and KV memory.

### Alignment and instruction tuning serve different purposes

Connector training helps the language side consume visual representations. Visual instruction data teaches question-conditioned use of images. Freezing the vision encoder, language model or selected modules changes cost and adaptation. A small trainable component does not remove the need to load frozen components.

If text frequently reveals answers, the model may rely on language priors. Replace an image with an unrelated one or shuffle image-text pairs and inspect whether answers change. Unchanged answers may indicate either a question that never required vision or a model ignoring visual evidence; distinguish those explanations.

### Seeing, interpreting and answering are separate stages

Reading a receipt total requires enough resolution to retain small print, recognition of numbers and layout, and selection of the total rather than subtotal or tax. A larger language model cannot recover evidence already discarded by preprocessing.

Identifying what a person on the left holds requires object recognition, spatial localization and attribute binding. Global semantic similarity is insufficient. Breaking errors into stages helps choose between resolution changes, visual modeling and improved supervision.

### Fluent hallucination can fill weak evidence

Language priors may fill gaps when visual evidence is blurred, compressed or absent, inventing objects, text or actions. Natural phrasing does not demonstrate evidential support.

Include questions about absent objects, ambiguous images and misleading textual suggestions. Allow uncertainty. Verify exact OCR and numerical answers directly, and check open descriptions against visible evidence. A prompt requesting no hallucinations is not a guarantee.

Start experiments with real preprocessing and CLIP/MAE outputs, then inspect projected shapes, visual-token counts and generation in a compatible VLM environment. The embedded small experiments establish the interface concepts; full GPU VLM inference requires appropriate resources. Finally, compare retrieval, classification, question answering and recommendation features on the same image. A representation useful for one objective need not solve the others.

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
