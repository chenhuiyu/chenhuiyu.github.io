---
title: "Reconstruction：遮住图像后，模型究竟学什么？"
date: "2026-09-10"
updated: "2026-09-10"
category: "Multimodal Understanding"
language: "zh-CN"
tags: ["multimodal", "Hands-on", "Model Learning"]
pairKey: "multimodal-reconstruction"
slug: "multimodal-reconstruction-zh"
excerpt: "区分像素重建与语义预测，拆解 MAE 的 visible-only encoder、轻量 decoder 和 masked loss。"
series: "multimodal"
seriesOrder: 3
draft: false
---

![Reconstruction：遮住图像后，模型究竟学什么？](/learning/multimodal-reconstruction-zh.svg)

## 遮掉输入，制造监督

Masked autoencoding 隐藏一部分 patch，让模型利用可见区域预测缺失内容。原始图像提供目标，不需要人工为每张图标注类别。MAE 的重要结构选择是 encoder 只处理可见 patch，decoder 再加入 mask tokens 和位置信息来还原完整顺序。下游任务常复用 encoder，而非把重建 decoder 当识别头。

196 个 patch、75% mask 时，可见 49 个。注意 CLS 和实现细节可能让实际 encoder 序列长度多一个。Encoder 序列缩短并不等于整个模型计算严格减少 75%，因为 decoder、投影与其他操作仍存在。

## Loss 只看哪些位置？

简化 masked MSE 为 $\sum_i m_i\|\hat p_i-p_i\|^2 / \sum_i m_i$，这里每个 patch 的平方误差先按内部维度平均，m=1 代表隐藏。不要与 attention mask 的 1=有效位置惯例混淆。预测全部 patch 但只监督 masked patch，有助于避免模型只复制输入。

```python
import torch
pred = torch.zeros(1, 4, 3)
target = torch.ones_like(pred)
mask = torch.tensor([[1.,0.,1.,0.]])
per_patch = ((pred-target)**2).mean(-1)
loss = (per_patch*mask).sum()/mask.sum()
assert loss.item() == 1.0
```

归一化 patch targets 会改变 loss 的尺度和重建显示方式。使用真实模型时读取 `norm_pix_loss` 等配置，不要直接把任意 logits 当作 0–255 像素。

## 重建漂亮不等于理解更好

像素目标会奖励纹理与低层细节；语义目标则可能希望忽略光照或背景。预测 latent representations 的方法改变监督空间，但要处理表示坍塌等问题。比较方法时分别测重建误差和迁移任务，不能用好看的补图证明检索或分类更强。

下方交互用合成灰度网格和常数预测器解释 masked loss，它没有训练过。真正的 `facebook/vit-mae-base` 重建在 Notebook：加载权重、固定随机 mask、显示原图/遮罩/重建，并打印模型自身 loss。

## 自测

**问题：** 把 mask ratio 设成 0，loss=0 是否说明模型完美？

<details><summary>展开答案</summary>不说明。没有被监督的隐藏位置，分母为 0；实际训练应拒绝这种设置或显式定义空损失。本页计算器返回 0 只是展示约定。</details>

## 本次真实运行结果

![真实 vit-mae-base 在合成色块上的重建；不是手工补图，也不代表自然图像效果。](/learning/multimodal-reconstruction-actual.png)

真实 vit-mae-base 在合成色块上的重建；不是手工补图，也不代表自然图像效果。

## 原始资料与继续阅读

- [Masked Autoencoders](https://arxiv.org/abs/2111.06377)
- [ViTMAE implementation](https://huggingface.co/docs/transformers/model_doc/vit_mae)
- [I-JEPA](https://arxiv.org/abs/2301.08243)

资料核对：2026-09-10。教学示例不代表生产基准；框架 API 和模型支持请以所链接版本为准。
