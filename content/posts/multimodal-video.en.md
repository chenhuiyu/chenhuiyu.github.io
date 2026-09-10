---
title: "Video understanding: sampling, temporal grounding, audio and evaluation"
date: "2026-09-10"
updated: "2026-09-10"
category: "Multimodal Understanding"
language: "en"
tags: ["multimodal", "Hands-on", "Model Learning"]
pairKey: "multimodal-video"
slug: "multimodal-video-en"
excerpt: "Treat video as temporal evidence, budget frames and tokens, and diagnose static shortcuts, subtitle leakage and missed events."
series: "multimodal"
seriesOrder: 5
draft: false
---

![Video understanding: sampling, temporal grounding, audio and evaluation](/learning/multimodal-video-en.svg)

## A complete explanation from first principles

### Video adds relationships across time

Objects move, actions have order, and sound can precede or follow an image. Answering whether someone closed a door before sitting down requires temporal comparison, not merely detecting doors and chairs. Some tasks need keyframes; others need continuous motion.

The model receives decoded, sampled and preprocessed frames rather than the original viewing experience. Frame rate, duration, resolution and sampling choices determine what evidence survives.

### Sampling sets an upper bound on observation

Sampling a sixty-second video at one frame per second produces sixty frames. A two-tenths-of-a-second event may be missed entirely. Higher rates improve coverage at additional token and compute cost. Coarse-to-fine or event-driven strategies introduce their own selection logic.

At 196 tokens per frame, sixty frames produce 11,760 visual tokens before text or audio. Real models may compress or use alternative spatiotemporal structures, but the arithmetic explains why naive frame accumulation becomes expensive.

### Preserve timestamps for localization

The tenth sampled frame is not necessarily the tenth second. Frame index, original timestamp and internal model position are different quantities. Variable-frame-rate videos and nonuniform sampling especially require an explicit mapping.

For a ground-truth interval [10,20] seconds and prediction [15,25], intersection length is five and union length fifteen, giving temporal IoU one-third. Recognizing that an event occurred and locating when it occurred are different tasks requiring different evaluation.

### Audio and subtitles can help or shortcut reasoning

Transcripts reveal speech, not necessarily visual facts. A subtitle mentioning a blue car does not prove that the visible car is blue. Silent frames likewise cannot establish a spoken name or an alarm sound.

Separate claims supported by images, audio and subtitles. Remove one modality at a time to test dependence on the expected evidence. Speech-recognition errors, dubbing mismatches and shifted subtitle timestamps can all corrupt fusion.

### Sliding windows need cross-window reasoning

A long video can be partitioned into windows whose representations or summaries are later combined. Overlap helps preserve boundary events but repeats computation. If one window shows lifting a cup and another lowering it, aggregation must still establish object identity and whether drinking occurred. Concatenating summaries does not solve that reasoning automatically.

Parallel windows can reduce wall-clock time without reducing total compute. Capacity, dependencies and final aggregation remain constraints. Streaming systems must distinguish immediate predictions from conclusions that require later frames.

### Test temporal understanding with controlled contrasts

Forward and reversed frame sequences contain the same objects while potentially expressing different action order. Check whether answers change appropriately. Add brief events, occlusions, scene cuts, conflicting audio and sparse evidence in long videos.

Record sampling policy, frame count, timestamp mapping, resolution, modalities and token budget, not duration alone. The notebook establishes single-frame patches and content representations. Extending those concepts to video adds the central problems of temporal evidence and resource budgeting.

## Multiple frames are not automatically temporal understanding

Encoding frames and concatenating them exposes multiple moments but does not necessarily teach order or duration. Tasks range from object recognition to event localization, ordering and causality. Preserve timestamps, sampling intervals and audiovisual alignment rather than treating a video as an unordered image bag.

A 60-second video at two frames per second yields 120 frames. At 256 retained tokens per frame, that is 30,720 visual tokens before text, audio or markers. Actual models may merge spatiotemporal patches; budget from real processor outputs.

## Sampling determines observable evidence

Uniform sampling is simple but can miss brief actions. Scene-change sampling covers transitions yet may miss subtle events in static scenes. Coarse-to-fine sampling identifies candidate windows before densifying them, adding calls and risking missed candidates.

```python
import numpy as np
sample_times = np.arange(0, 10, 1.0)
event_start, event_end = 4.2, 4.4
hits = (sample_times >= event_start) & (sample_times <= event_end)
print('observed event?', hits.any())  # False
```

A stronger downstream model cannot recover unobserved evidence reliably. Audio or ASR may supply missing information, but transcripts can also create shortcuts that bypass vision.

## Temporal grounding and evidence ablations

Temporal IoU is interval intersection divided by union duration. Report recall at stated thresholds alongside boundary errors. Shuffled frames, single-frame inputs and removal of audio/subtitles test which evidence is used. Reversing a video should affect an order-sensitive answer appropriately.

Split data by source video, creator or event to avoid adjacent clips crossing train/test boundaries. Slice evaluation by language, duration, resolution, small objects and OCR. Averages can hide systematic weaknesses.

## Check your understanding

**Question:** More frames improve accuracy. Does that prove stronger temporal reasoning?

<details><summary>Answer</summary>Not alone. Sampling may simply capture a key frame more often. Control evidence coverage and add order-sensitive tasks and frame-permutation ablations.</details>

## Primary sources and further reading

- [Qwen2.5-VL](https://arxiv.org/abs/2502.13923)
- [Video-MME](https://arxiv.org/abs/2405.21075)
- [VideoMAE](https://arxiv.org/abs/2203.12602)

Sources checked on 2026-09-10. Teaching examples are not production benchmarks; confirm APIs and model support against the linked version.
