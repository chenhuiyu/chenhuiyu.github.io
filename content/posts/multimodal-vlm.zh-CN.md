---
title: "视觉语言模型：从 CLIP 特征到能回答问题的 LLM"
date: "2026-09-10"
updated: "2026-09-10"
category: "Multimodal Understanding"
language: "zh-CN"
tags: ["multimodal", "Hands-on", "Model Learning"]
pairKey: "multimodal-vlm"
slug: "multimodal-vlm-zh"
excerpt: "理解 vision encoder、projector 与 decoder 的接口，分清对齐训练、指令训练和视觉 grounding。"
series: "multimodal"
seriesOrder: 4
draft: false
---

![视觉语言模型：从 CLIP 特征到能回答问题的 LLM](/learning/multimodal-vlm-zh.svg)

## 两种模型，输出目标不同

CLIP 通常给匹配分数；视觉语言生成模型需要根据图像与问题输出文本。一个常见结构是 vision encoder 提取 `[B,N,Dv]`，projector 映射到语言 hidden dimension Dl，视觉表示再按模型协议插入语言序列。Projector 解决接口维度与表示匹配的一部分问题，不是简单“翻译图片成句子”。

具体模型可能采用 cross-attention、resampler 或视觉 token 合并，不能把所有 VLM 都简化为单个线性层。图像 token、special tokens 和位置编码必须与 processor/chat template 配套。

## 训练通常不止一个阶段

对齐阶段可以冻结某些 backbone，只训练桥接模块；视觉指令训练用图像、问题、回答监督生成。哪些模块更新依论文与 recipe 而定。模型会利用语言先验：如果训练问题里“香蕉是什么颜色”的答案几乎都是黄色，它可能不看图也答对。因此需要设计图像改变但问题相同的对照。

```python
import torch
visual = torch.randn(1, 196, 768)
projector = torch.nn.Linear(768, 1024)
visual_tokens = projector(visual)
text_tokens = torch.randn(1, 12, 1024)
combined = torch.cat([visual_tokens, text_tokens], dim=1)
assert combined.shape == (1, 208, 1024)
```

这段只演示 shape 接口，不是完整 VLM：缺 processor、模态标记、position/mask 以及预训练对齐。

## Grounding 要有空间证据

“图里有猫”与“猫在红色沙发左侧，框坐标是多少”是不同能力。坐标还可能用像素、归一化 0–1 或离散 bins，resize/crop 后需要映射回原图。OCR 要保留足够分辨率；细粒度属性需要局部证据。语言流畅不能代替这些检查。

读 Qwen2.5-VL 等报告时，重点看动态分辨率、视觉 token 预算和视频时间表示如何连接任务。不要把报告中的最大上下文理解为任何设备都能高效运行。

## 自测

**问题：** 遮住图像后准确率不变，说明视觉 encoder 很鲁棒吗？

<details><summary>展开答案</summary>更应怀疑模型利用问题或数据集先验。比较图片打乱、空图、问题-only 与反事实图像，验证答案是否依赖视觉证据。</details>

## 原始资料与继续阅读

- [LLaVA / Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
- [Qwen2.5-VL report](https://arxiv.org/abs/2502.13923)

资料核对：2026-09-10。教学示例不代表生产基准；框架 API 和模型支持请以所链接版本为准。
