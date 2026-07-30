---
title: "从商品向量到合法 Semantic ID｜生成式推荐 Hands-on Lab"
date: "2026-07-30"
updated: "2026-07-30"
category: "Generative Recommendation"
language: "zh-CN"
tags:
  - Generative Recommendation
  - Semantic ID
  - Residual Quantization
  - Beam Search
  - Python
pairKey: "generative-recommendation-hands-on"
slug: "generative-recommendation-hands-on-zh"
excerpt: "一个可以亲手操作的生成式推荐实验：观察残差量化如何得到 Semantic ID，调节解码参数，对比 Trie 约束，并在浏览器内真实运行 Python。"
series: "generative-recommendation"
seriesOrder: 21
draft: false
---

读完 TIGER、LETTER 或 OneRec，最容易留下两个错觉：Semantic ID 只是把商品编号换了一种写法；beam search 也不过是“多保留几个候选”。真正把代码跑一遍，会发现困难恰好藏在这两句话省略的部分——向量为什么能被离散化、离散地址怎样进入生成目标、合法 token 为什么仍能拼成不存在的商品，以及最后究竟该评估推荐质量还是生成质量。

这份 tutorial 把整条链路压缩到一个可以从头运行的教学系统里。它与「20 篇论文读懂生成式推荐」共用同一条叙事主线，但不要求你先读完所有文章。

## 先选一种运行方式

你可以按自己的目的选择三种入口：

| 入口 | 适合什么 | 你能改什么 |
|---|---|---|
| 本页交互实验 | 先建立直觉，几分钟走完整条链路 | 商品、temperature、beam width、decode depth、Trie 开关与 Python |
| [下载完整 Notebook](/notebooks/generative-recommendation-hands-on.ipynb) | 本地 Jupyter / VS Code 中逐格实验 | BPR 训练、codebook、Semantic ID、解码器与评测 |
| [在 Colab 打开](https://colab.research.google.com/github/chenhuiyu/chenhuiyu.github.io/blob/source/public/notebooks/generative-recommendation-hands-on.ipynb) | 不安装环境，复制一份到自己的 Google Drive | Notebook 中的全部代码与参数 |

本页实验只使用八件商品，强调交互反馈；Notebook 使用 12 件商品和八条用户历史，会真正训练一个小型 BPR、拟合多层 residual codebook，再运行有约束和无约束的生成与评测。两者讲的是同一件事，只是放大倍数不同。

## 实验地图：它和系列文章怎样对应

| 实验阶段 | 系列中的对应章节 | 这一阶段要回答的问题 |
|---|---|---|
| 连续打分 | [01 · BPR](/blog/generative-recommendation-01-bpr-zh) | 推荐系统原本怎样学习用户与商品向量？ |
| 序列变成 token | [02 · GRU4Rec](/blog/generative-recommendation-02-gru4rec-zh)、[03 · SASRec](/blog/generative-recommendation-03-sasrec-zh)、[04 · BERT4Rec](/blog/generative-recommendation-04-bert4rec-zh) | 为什么用户历史不能再被压成一个静态点？ |
| 检索变成生成 | [10 · DSI](/blog/generative-recommendation-10-dsi-zh)、[11 · TIGER](/blog/generative-recommendation-11-tiger-zh) | 模型怎样直接写出目标的离散地址？ |
| 设计 Semantic ID | [13 · LC-Rec](/blog/generative-recommendation-13-lc-rec-zh)、[14 · LETTER](/blog/generative-recommendation-14-letter-zh)、[15 · ETEGRec](/blog/generative-recommendation-15-etegrec-zh) | 好 ID 应该保留内容、协同信号，还是服务最终推荐损失？ |
| 统一生成 | [17 · OneRec](/blog/generative-recommendation-17-onerec-zh) | 召回与排序能否共享同一个生成目标？ |
| 失败与泛化 | [2026 · CARE](/blog/generative-recommendation-2026-02-care-zh)、[2026 · 泛化审计](/blog/generative-recommendation-2026-07-generalization-zh) | 深层误差、非法 ID 与 unseen item 应该怎样测？ |

## 0. 我们到底在优化什么

设用户历史为 \(x_{1:t}\)，传统打分模型为每个候选商品计算 \(s(u,i)\)，再从目录中取 Top-K。生成模型则把目标商品写成 \(M\) 个离散 token：

$$
i \longrightarrow (c_1,c_2,\ldots,c_M), \qquad
p(i\mid x_{1:t})=\prod_{m=1}^{M}p(c_m\mid x_{1:t},c_{<m})
$$

这个改写带来三个新接口：

1. **Tokenizer**：连续商品向量怎样变成稳定、有区分度的离散地址；
2. **Decoder**：用户条件怎样影响每一层 token 的概率；
3. **Catalog constraint**：怎样保证生成的完整地址确实对应商品。

Notebook 会把三个接口分开实现，避免“一个大模型把一切都做了”掩盖真正的工程边界。

## 1. 从 BPR 得到商品向量

Notebook 先把每条用户历史的最后一件商品留作 next-item 检查，再使用其余隐式反馈训练一个 NumPy 版 BPR。对用户 \(u\)、正样本 \(i\) 和负样本 \(j\)，目标是：

$$
\mathcal{L}_{\text{BPR}}
=-\log \sigma\big(s(u,i)-s(u,j)\big)+\lambda\lVert\Theta\rVert_2^2
$$

你会看到 pairwise loss 下降、每个用户的 Top-3 结果，以及商品 embedding 的二维投影。这里最重要的不是 toy Recall@3 数字，而是确认后续 tokenizer 接到的是**由推荐行为学习出来的连续表示**。

对应系列：[BPR](/blog/generative-recommendation-01-bpr-zh) 解释打分目标；[LC-Rec](/blog/generative-recommendation-13-lc-rec-zh) 则追问只用内容语义是否足够。

## 2. 用 Residual Quantization 得到 Semantic ID

第 \(m\) 层 codebook 只解释上一层留下的残差：

$$
q_m=\arg\min_k\lVert r_{m-1}-e_{m,k}\rVert_2^2,\qquad
r_m=r_{m-1}-e_{m,q_m}
$$

四层 codebook 最终给每件商品一个四 token 地址。前缀表达粗粒度区域，后续 token 逐步修正重构。Notebook 会同时输出：

- 每件商品的完整 Semantic ID；
- 每增加一层后的平均 residual norm；
- ID 碰撞率；
- 每层 token 使用分布。

低重构误差并不自动等于好推荐。两个商品可能在向量空间很近，却属于不同协同人群；一个 codebook 也可能重构很好，却把大量商品挤进少数 token。[TIGER](/blog/generative-recommendation-11-tiger-zh)、[LETTER](/blog/generative-recommendation-14-letter-zh) 和 [DIGER](/blog/generative-recommendation-2026-01-diger-zh) 的区别，正是在“谁来定义好 ID”。

## 3. 把推荐改写成下一 token 生成

页面里的可视化使用固定 logits，让你能立即调节参数。Notebook 则根据生成出的 Semantic ID 统计一个带 smoothing 的 toy token model。它不是 Transformer，但保留了自回归解码最关键的计算：

$$
\log p(c_{1:M})=\sum_{m=1}^{M}\log p(c_m\mid c_{<m},x)
$$

- **Temperature** 改变分布尖锐程度；
- **Beam width** 决定每层保留多少条前缀；
- **Decode depth** 让你观察误差怎样沿层级累积；
- **完整前缀** 而不是单个 token，才是目录中的真实地址。

这一步连接 [DSI](/blog/generative-recommendation-10-dsi-zh) 的“生成即检索”、[GPTRec](/blog/generative-recommendation-12-gptrec-zh) 的列表生成与 [OneRec](/blog/generative-recommendation-17-onerec-zh) 的统一生成目标。

## 4. 为什么 Trie 约束不是可选装饰

假设每一位 token 都来自合法词表，完整序列仍可能不对应任何商品。无约束模型会把不同商品的高概率局部片段拼在一起，得到一个目录外 ID。Trie 在每一步只保留至少能通向一件真实商品的前缀：

$$
\mathcal{A}(c_{<m})=
\{c\mid \exists i,\ \operatorname{SID}(i)_{1:m}=c_{<m}\oplus c\}
$$

页面允许你直接关闭约束；Notebook 会把 beam 中合法和非法的地址画成两种颜色，并对多个 temperature 采样。默认实验通常会看到无约束采样的大量地址落在目录外，而 Trie 约束后的 invalid-ID rate 精确为零。

注意：Trie 只能保证“商品存在”，不能保证“商品适合这个用户”。它是解码合法性层，不是推荐相关性模型。

## 5. 评测不要只留一个 Recall@K

一个完整实验至少要把四组指标拆开：

| 层面 | 最小指标 | 它防止什么误判 |
|---|---|---|
| 推荐质量 | Recall@K、NDCG@K、分人群/头尾商品结果 | 模型能生成，不等于推荐得准 |
| Tokenizer | 重构误差、碰撞率、codebook usage、前缀负载 | 向量重构漂亮，不等于 ID 可用 |
| 解码 | invalid-ID rate、catalog coverage、延迟、beam 成本 | Top-1 看似正常，长尾 beam 可能已经失控 |
| 泛化 | unseen-item、冷启动、时间外推、目录变更 | 记住训练目录，不等于会泛化 |

Notebook 的最后几个 cell 会运行可失败的 assertions：BPR loss 必须下降、残差必须缩小、Semantic ID 长度必须一致、Trie 输出不能出现非法 ID。它们很小，但比只展示一张漂亮图更接近可复现研究。

## 6. 推荐的动手顺序

第一次先保持默认参数完整运行，然后一次只改一件事：

1. 把 temperature 改成 `0.4` 与 `1.4`，比较非法率和 catalog coverage；
2. 把 beam width 改成 `4` 与 `32`，观察更多候选是否只是暴露更多非法组合；
3. 把 codebook size 改成 `3` 或 `6`，记录重构误差、碰撞与 token 使用率；
4. 把 codebook 层数改成 `2` 或 `5`，观察深度与误差累积；
5. 最后把 BPR embedding 换成内容 encoder 或序列模型输出。

第 3、4 步对应 LETTER 与 CARE；第 5 步会把你带向 LLaRA、HSTU、ContRec 与 DIGER。

## 7. 这份教程没有假装复现什么

本页与 Notebook 都是机制实验，不是 TIGER、OneRec 或 DIGER 的 benchmark 复现。它们没有真实规模的序列 Transformer、端到端 tokenizer 学习、负载均衡损失、分布式 beam search，也没有线上延迟和 A/B 测试。

它们真正提供的是一条可检查的骨架：每一个中间表示都能打印，每一个解码选择都能改，每一个合法性假设都能被关闭。下面先用页面实验建立直觉；需要完整代码路径时，再进入 Notebook。
