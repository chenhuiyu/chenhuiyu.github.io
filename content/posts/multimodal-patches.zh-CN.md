---
title: "图像怎样变成 token：Patch、位置与分辨率"
date: "2026-09-10"
updated: "2026-09-10"
category: "Multimodal Understanding"
language: "zh-CN"
tags: ["multimodal", "Hands-on", "Model Learning"]
pairKey: "multimodal-patches"
slug: "multimodal-patches-zh"
excerpt: "从 RGB 张量到 ViT patch embedding，计算视觉 token 数，理解缩放与裁剪可能丢失的内容。"
series: "multimodal"
seriesOrder: 1
draft: false
---

![图像怎样变成 token：Patch、位置与分辨率](/learning/multimodal-patches-zh.svg)

## 图像不是一长串像素文字

视觉 encoder 常把图片分成固定大小 patch，每块展平后线性投影成 D 维向量。设输入 `[B,3,H,W]`，patch 边长 P，整除时 token 数 $N=(H/P)(W/P)$，每块原始维度为 $3P^2$。这是可学习的线性变换，不是 tokenizer 按词表查找离散 ID。某些架构添加 CLS token，另一些使用 pooling 或压缩，需读具体实现。

224×224 图像配 16×16 patch 得到 196 个 patch，若有 CLS 则是 197。分辨率变成 448×448，patch 数是 784，dense attention 的 pair 数约变为原来的 16 倍，不只是图像边长的 2 倍。

## 一段能运行的 patchify

```python
import torch
image = torch.arange(3*32*32).reshape(1,3,32,32).float()
p = 8
patches = image.unfold(2,p,p).unfold(3,p,p)
patches = patches.permute(0,2,3,1,4,5).reshape(1,16,3*p*p)
projection = torch.nn.Linear(3*p*p, 64)
print(projection(patches).shape)  # [1,16,64]
```

`permute` 的顺序决定像素排列，重建时必须用一致的逆变换。Tensor shape 正确并不能保证颜色通道和空间位置正确。Notebook 会把 patchify / unpatchify 往返结果与原图比较。

## 预处理是模型定义的一部分

Resize、crop、RGB conversion、归一化 mean/std 都会影响结果。把已经除以 255 的浮点图再次交给默认 rescale processor，可能重复缩放。把长图裁成方形则可能直接丢失边缘文字；生成器回答错误前，先检查送入 encoder 的图。

动态分辨率模型可能按长宽比例生成不同视觉 token 数，视频还引入时间维度。不要把固定 ViT 的 196-token 经验搬到所有 VLM。

## 自测

**问题：** 把 patch 边长从 16 改成 8，可以直接沿用原 patch projection 权重吗？

<details><summary>展开答案</summary>通常不行。展平维度从 768 变为 192，权重 shape 已改变；位置表示与训练分布也会受影响。架构适配需要明确方法，而非只改配置整数。</details>

## 原始资料与继续阅读

- [Vision Transformer](https://arxiv.org/abs/2010.11929)
- [Image processors](https://huggingface.co/docs/transformers/main_classes/image_processor)

资料核对：2026-09-10。教学示例不代表生产基准；框架 API 和模型支持请以所链接版本为准。
