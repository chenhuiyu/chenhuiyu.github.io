---
title: "一个模型承包召回、排序、解释和创作｜生成式推荐 06：M6-Rec"
date: "2026-07-29"
updated: "2026-07-29"
category: "Generative Recommendation"
language: "zh-CN"
tags:
  - Generative Recommendation
  - Recommender Systems
  - M6-Rec
  - Foundation Models
pairKey: "generative-recommendation-06-m6-rec"
slug: "generative-recommendation-06-m6-rec-zh"
excerpt: "M6-Rec 把统一推荐推进到真实工业链路：召回、CTR、解释、对话和内容创作共享语言底座，再用 option tuning、late interaction 与蒸馏控制成本。"
series: "generative-recommendation"
seriesOrder: 6
draft: false
---

“一个模型做所有推荐任务”听起来很优雅，直到你把它放进真实 App。

召回要从百万商品里找几百件；CTR 排序只有几十毫秒；解释和对话需要生成文本；手机端又只有几 MB 的模型预算。一个大模型即使离线指标很好，只要延迟太高、每个任务都要复制一份参数，就仍然无法成为工业底座。

[上一篇 P5](/blog/generative-recommendation-05-p5-zh) 证明了五类推荐任务可以共享 text-to-text 接口。[M6-Rec: Generative Pretrained Language Models are Open-Ended Recommender Systems](https://arxiv.org/abs/2205.08084) 接着追问：

> 统一模型能不能覆盖召回、排序、解释、对话与内容创作，并且同时部署在云端和手机？

M6-Rec 的答案不是一个单独算法，而是一套“底座 + 任务形式 + 适配 + 加速”的系统设计。它把 foundation model 从论文概念拉进了推荐系统的性能预算表。

## 30 秒看懂本文

1. **P5 留下的问题**：任务在接口上统一了，但全目录召回、实时排序、边缘部署等工业约束没有被解决。
2. **M6-Rec 的答案**：把用户行为和商品信息写成文本，以 M6 为共享底座；检索用对比学习向量，排序用语言理解，开放任务用生成，并以 option-adapter tuning、late interaction、蒸馏、剪枝和早退降低每任务参数与延迟。
3. **最重要的证据**：M6-Rec 在 TaoProduct CTR 上把 DIN 的 AUC 从 0.7611 提到 0.7995；MiniApp 召回 HR@100 从 TwinBERT 的 69.6% 提到 74.1%；线上替换召回模型带来超过 1.0% 的相对 CTR 提升。

![从学术统一到工业开放任务](/blog/generative-recommendation/06-m6-rec/paradigm-shift.svg)

*图 1：工业 foundation model 的价值函数同时包含任务覆盖、样本效率、延迟、模型大小与部署位置。*

## 上一篇留下了什么问题

P5 的统一建立在一个 encoder–decoder 上，但不同产品环节的计算形态并不统一：

| 环节 | 典型输入 | 典型输出 | 关键约束 |
|---|---|---|---|
| 召回 | 用户历史 | 近邻向量 / 候选集 | 亿级目录、可建 ANN 索引 |
| CTR 排序 | 用户 + 候选 | 点击概率 | 低延迟、高吞吐 |
| 解释 | 用户 + 商品 | 文本 | 忠实、可读 |
| 对话 | 多轮上下文 | 回复与商品 | 连贯、可控 |
| 内容创作 | 用户群或行为 | 查询、标题、图片提示 | 多样性、质量 |

如果所有任务都强行用自回归 beam search，召回会慢得不可用；如果每个下游任务完整 fine-tune 一份底座，存储和维护又会爆炸。

M6-Rec 的第一条工程原则是：

> 共享的是预训练知识与大部分参数，不必共享完全相同的输出头和在线执行路径。

这比“所有任务都生成文本”更务实。

## 用八件商品理解开放任务

仍然使用小林的行为：

```text
I01 网球拍 → I03 攀岩鞋 → I05 羽毛球 → I06 泳镜
```

在 M6-Rec 中，它会被改写为类似：

```text
一名年轻男性用户，最近点击了网球拍、攀岩鞋和羽毛球……
```

论文刻意避免把 Item ID 作为核心输入，主要依赖标题、类目和描述等文本。这让新查询、新商品即使没有训练期 ID embedding，也能被编码。

同一份文本可以进入四种路径：

1. **召回**：把用户文本编码成向量 $\mathbf x$，把每件商品文本编码成 $\mathbf y$，用 kNN 找候选；
2. **排序**：拼接历史与候选 `I06 泳镜`，预测点击 / 不点击；
3. **解释**：生成“因为你最近关注球类和水上运动……”；
4. **创作**：根据行为生成可能点击的新搜索词或商品标题。

![同一份用户文本贯穿推荐漏斗](/blog/generative-recommendation/06-m6-rec/toy-open-tasks.svg)

*图 2：底座共享不等于头部完全相同。检索需要可索引向量，排序需要概率，开放任务需要 token 序列。*

这个设计也暴露一个取舍。文本能泛化到未见商品，却可能丢失匿名协同信号：两件标题毫不相似的商品也许被同一群用户连续购买。M6-Rec 在论文里优先选择语义泛化；后面的 LLaRA 会专门补回协同行为。

## 模型从输入到输出发生了什么

### 1. 共享预训练底座

M6 是类似 GPT/T5 的工业预训练语言模型，使用 text infilling 与自回归生成目标。M6-Rec 不从零训练推荐大模型，而是把推荐数据格式化为普通文本后适配这个底座。

### 2. 检索：双塔向量与对比学习

用户文本与商品文本分别经过 Transformer。指定位置的表示被线性投影到 128 维并做 $L_2$ 归一化：

$$
\mathbf x,\mathbf y\in\mathbb R^{128},
\qquad
\|\mathbf x\|_2=\|\mathbf y\|_2=1.
$$

线上可以预计算所有商品向量并建立 kNN 索引，而不必对每件商品做自回归解码。

### 3. 排序：multi-segment late interaction

完整大模型逐候选运行太慢。M6-Rec 把请求拆成细粒度 segment：

```text
性别画像 | 点击 I01 | 点击 I03 | 点击 I05 | 候选 I06
```

前 $L'$ 层分别编码并缓存每段；请求到来时，拼接缓存表示，只运行最后 $L-L'$ 层做跨段交互。论文的一个配置是前 21 层预计算、线上运行最后 3 层。

细粒度分段很重要。用户新增一次点击时，只需为新行为计算一个 segment，旧点击和候选商品的深层表示可以复用。

### 4. 适配：option tuning 与 adapter

普通 prompt tuning 在输入前放一组可训练 soft prompt，却仍额外学习分类层。M6-Rec 把最后 $C$ 个 soft prompt 直接复用为 $C$ 个类别的“soft options”，例如点击 / 不点击。

再在每层 FFN 加低秩 adapter，只训练约 1% 的任务参数。Table 8 中，option-adapter tuning 在三个 CLUE 任务上甚至略高于 full fine-tuning。

### 5. 压缩：手机端 M6-Edge

论文把约 300M 的 M6-base 蒸馏为 10M 的 M6-Edge，再通过参数共享与剪枝得到 2M 版本；还为不同层加入 early-exit loss，使设备可按预算提前退出。

![M6-Rec 的 late-interaction 服务流程](/blog/generative-recommendation/06-m6-rec/mechanism.svg)

*图 3：大部分深层计算前置并缓存，在线只保留少量跨段交互。加速改变了计算位置，而不是简单删掉交互。*

## 核心公式逐项拆解

召回使用温度化对比损失：

$$
\mathcal L_{\text{retrieval}}
=
-
\sum_{\langle x,y\rangle}
\log
\frac{\exp(\mathbf x^\top\mathbf y/\tau)}
{\exp(\mathbf x^\top\mathbf y/\tau)
+
\sum_{\mathbf y'\in Y'}
\exp(\mathbf x^\top\mathbf y'/\tau)}.
$$

- $\langle x,y\rangle$：真实发生过点击的用户—商品正对；
- $\mathbf x^\top\mathbf y$：归一化后的相似度；
- $Y'$：同 batch 中采样的负商品；
- $\tau=0.07$：论文使用的温度，控制 softmax 的尖锐程度。

对小林而言，$\mathbf x$ 来自“点击网球拍、攀岩鞋、羽毛球”的文本，$\mathbf y$ 来自“泳镜”的标题和属性。训练希望正对相似度高于瑜伽垫、蛋白棒等负对。

这条公式说明 M6-Rec 并没有让“召回”变成逐 token 生成。它让语言模型成为语义编码器，再回到工业成熟的 ANN 检索结构。

## 训练和推理分别怎么做

**召回训练**

```text
user_vec = normalize(project(M6(user_text)))
item_vec = normalize(project(M6(item_text)))
negatives = other items in the mini-batch
loss = contrastive_softmax(user_vec, item_vec, negatives, tau=0.07)
```

**召回推理**

```text
offline:
    encode every item and build ANN index
online:
    encode current user
    retrieve nearest item vectors
```

**CTR 排序**

```text
offline/cache:
    encode each stable segment through layers 1...L'
online:
    combine cached user/history/candidate segments
    run layers L'+1...L
    score click option
```

**开放生成**

```text
format behavior as text
adapt the shared model with generation examples
decode explanation, dialogue, query, or title
```

因此，M6-Rec 是“一套共享底座上的多种执行计划”，而不是一个函数强行替代整条推荐链路。

## 实验究竟证明了什么

离线效果：

- TaoProduct CTR：DIN 0.7611，M6-Rec **0.7995**；
- AlipayQuery CTR：DIN 0.7332，M6-Rec **0.7508**；
- AlipayMiniApp HR@100：TwinBERT 69.6%，M6-Rec **74.1%**；
- 对训练未见商品：TwinBERT 49.6%，M6-Rec **57.0%**；ID-based YouTubeDNN 无法处理；
- explainable recommendation 的 BLEU-4：PETER+ 3.06，M6-Rec **3.59**。

服务代价：

- 完整 24 层 M6-Rec：AUC 0.7995，57 ms；
- 蒸馏 3 层模型：AUC 0.7566，16 ms；
- late interaction、线上 3 层：AUC **0.7731**，16 ms。

![M6-Rec 的效果—延迟证据](/blog/generative-recommendation/06-m6-rec/evidence.svg)

*图 4：数值来自原论文 Tables 2、3、6。延迟组为方便同图展示按 100 归一化，标签保留真实毫秒。*

late interaction 与 3 层蒸馏模型延迟相同，却保留更多 AUC，说明缓存深层表示比直接把整个模型削薄更能保住能力。但它仍比完整模型低 0.0264 AUC，不能把加速写成“零损失”。

线上证据更难得：

- MiniApp 召回替换 TwinBERT-like 基线后，相对 CTR 提升超过 **1.0%**，并于 2021 年 7 月全量部署；
- 手机端 M6-Edge ranker 带来约 **0.4%** 点击提升。

这些数字来自论文报告，不是独立复现；线上实验的流量、时长和置信区间没有完整公开。

## 它失败在哪里

### 1. “Open-ended” 是能力范围，不是无限任务保证

论文覆盖多类任务，但每个任务仍需数据格式、prompt、head 或适配参数。一个从未定义输出协议的新业务并不会自动可用。

### 2. 去掉 ID 提升冷启动，也损失协同辨识度

文本相似不等于同一群人会购买。两件语义接近的商品可能服务完全不同的价格带和用户群；两件语义不同的商品也可能强共现。

### 3. late interaction 有缓存系统成本

缓存失效、segment 版本、用户隐私、内存占用和热点候选更新都是额外工程。论文报告模型延迟，不等于端到端系统延迟。

### 4. 任务间仍有专用组件

检索使用投影层与 ANN，分类使用 soft options，生成使用 decoder。M6-Rec 统一了底座，但没有消灭推荐漏斗。

### 5. 生成内容的忠实性未被充分证明

BLEU、ROUGE 和多样性不保证解释真实反映模型决策，也不保证生成商品标题安全、合规、可销售。

### 6. 论文难以完整复现

关键数据来自支付宝、淘宝内部，M6 预训练语料、在线系统和部署环境不是公开实验室可以等价复刻的。

## 下一篇为什么会出现

P5 和 M6-Rec 都建立在一个乐观前提上：大语言模型已经拥有丰富知识，只需把推荐任务写成合适语言形式，就能迁移能力。

但如果直接把用户历史和目标商品交给通用 LLM，会发生什么？

[下一篇 TALLRec](/blog/generative-recommendation-07-tallrec-zh) 给出了一个不太浪漫的答案：零样本 LLM 在推荐 AUC 上接近随机猜测。语言能力和偏好判断之间存在 domain gap；少量 recommendation instruction tuning，而不是更花哨的 prompt，才真正把它拉过来。

---

**论文**：Cui et al., “M6-Rec: Generative Pretrained Language Models are Open-Ended Recommender Systems,” 2022.

**原文**：[arXiv:2205.08084](https://arxiv.org/abs/2205.08084)

**系列导航**：[二十篇论文看懂生成式推荐的前世今生](/series/generative-recommendation)
