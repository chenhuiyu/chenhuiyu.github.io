---
title: "Top-K 不是唯一答案，列表也可以逐件写｜生成式推荐 12：GPTRec"
date: "2026-07-29"
updated: "2026-07-29"
category: "Generative Recommendation"
language: "zh-CN"
tags:
  - Generative Recommendation
  - Recommender Systems
  - GPTRec
  - List Generation
pairKey: "generative-recommendation-12-gptrec"
slug: "generative-recommendation-12-gptrec-zh"
excerpt: "GPTRec 对比一次打分的 Top-K 与逐件生成的 Next-K，并用 SVD 量化把每件商品表示成多个紧凑 token。"
series: "generative-recommendation"
seriesOrder: 12
draft: false
---

推荐系统最后交付的不是一个商品，而是一张列表。

传统 Top-K 的做法是给每个候选独立打分，再取最高的 K 个。这样很快，却有一个结构性盲点：排第二的商品不知道第一是谁。两双几乎相同的鞋可以同时高分，模型却没有机会在生成第二件时说“第一件已经覆盖跑步了，我换一个攀岩相关商品”。

[GPTRec: Generative Sequential Recommendation with GPTRec](https://arxiv.org/abs/2306.11114) 把这件事拆成两个问题：

1. 商品一定要“一件一个 token”吗？
2. 推荐列表一定要“一次 Top-K”吗？

它给出的答案分别是 SVD 多 token 表示，以及逐件生成的 Next-K。

## 30 秒看懂本文

1. **TIGER 留下的问题**：Semantic ID 解决了一个商品如何生成，却没有直接建模列表成员之间的关系。
2. **GPTRec 的答案**：使用 GPT-2 式 decoder 建模历史；Top-K 一次输出 K 个独立分数，Next-K 则学习 \(\prod_kP(r_k\mid H,r_{<k})\)，让后面的推荐看到前面的选择。
3. **最重要的结果**：MovieLens-1M 上 GPTRec-TopK 的 NDCG@10 为 0.146，接近 BERT4Rec 的 0.152；NextK 为 0.105。四个 SVD token、每维 512 个值时只需 2048 个 embedding，NDCG@10 为 0.108。

先澄清一个容易被名字误导的事实：

> GPTRec 使用 GPT-2 的 decoder-only 架构，但论文没有载入预训练 GPT-2 checkpoint。“GPT”在这里主要指自回归架构，不是调用通用大模型知识。

![Top-K 与 Next-K 的差别](/blog/generative-recommendation/12-gptrec/paradigm-shift.svg)

*图 1：Top-K 的候选分数可并行计算；Next-K 的每一步以已经推荐的列表前缀为条件。*

## 用八件商品理解列表条件

用户最近行为仍是：

```text
I01 网球拍 → I03 攀岩鞋 → I05 羽毛球拍 → I07 泳帽
```

Top-K 可能独立给出：

```text
I06 泳镜 0.92
I08 运动毛巾 0.89
I02 网球 0.87
```

Next-K 则生成：

```text
第 1 步：P(I06 | 历史)
第 2 步：P(I08 | 历史, I06)
第 3 步：P(I04 | 历史, I06, I08)
```

第二步可以利用“已经有泳镜”这个事实；第三步可以主动覆盖攀岩兴趣。**模型结构提供了列表交互的通道**，但这不代表模型自动学会多样性——训练目标若仍只是普通 next-item likelihood，它可能继续复制受欢迎或相似商品。

![八件商品如何逐件组成列表](/blog/generative-recommendation/12-gptrec/toy-next-k.svg)

*图 2：Next-K 具备表达列表策略的条件结构；真正优化多样性还需要相应数据、目标或强化学习信号。*

## 商品 token 的第二条路线：SVD 量化

TIGER 从内容 embedding 学 Semantic ID。GPTRec 提出一种更直接的协同压缩方式：先分解用户—商品交互矩阵。

令 \(M\in\mathbb R^{|U|\times|I|}\)，截断 SVD 为：

\[
M\approx U\Sigma E^\top.
\]

\(E_i\in\mathbb R^t\) 是商品 \(i\) 的 \(t\) 维协同坐标。对每一维：

1. 把所有商品值归一化到 \([0,1]\)；
2. 加标准差 \(10^{-5}\) 的微小高斯噪声，打破边界并列；
3. 量化成 \(v\) 个桶；
4. 第 \(d\) 维的 token 区间整体偏移 \(v(d-1)\)，避免不同维共享编号。

若 \(t=4,v=512\)，每件商品变成四个 token，而总 embedding 只有 \(4\times512=2048\) 个，与商品数无关。

```text
I06 泳镜 → 〈维1-117, 维2-403, 维3-026, 维4-288〉
```

这不是内容语义，而是交互矩阵的离散坐标。两件被相似用户消费的商品会得到相近量化模式。

## 核心公式：从独立集合到有序联合分布

Next-K 的核心不是网络层，而是分解方式。给定历史 \(H\)，推荐列表 \(R=(r_1,\ldots,r_K)\) 的概率为：

\[
P(R\mid H)
=\prod_{k=1}^{K}
P(r_k\mid H,r_{<k}).
\]

- \(r_1\) 只看历史；
- \(r_2\) 同时看历史和第一件推荐；
- \(r_{<k}\) 是已经写出的列表前缀；
- 顺序因此成为模型的一部分。

若用一件一个 token，softmax 直接覆盖商品目录；若用 SVD token，每个 \(r_k\) 又展开为 \(t\) 个子 token。后者省 embedding，却让一次列表生成需要更多自回归步骤。

![GPTRec 的 token 与列表机制](/blog/generative-recommendation/12-gptrec/mechanism.svg)

*图 3：商品压缩与列表生成是两个正交设计轴；可以一 token + Next-K，也可以多 token + Top-K。*

## 训练与推理怎么做

### Top-K 模式

输入用户历史，读取最后一个 hidden state，一次与全部 item embedding 计算 logits，取 K 个最高分。只需一次 scoring pass，和 SASRec 类似。

### Next-K 模式

把已输出商品追加到上下文，再预测下一件。取 10 件就需要约 10 次 scoring pass；如果商品本身由多个 token 表示，步骤更多。

```text
context = user_history
for k in 1..K:
    next_item = decode_one_item(context)
    output.append(next_item)
    context.append(next_item)
```

论文的训练仍以历史中的真实下一商品为主，并没有提供成熟的列表级 RL 或多样性监督。这一点很关键：**Next-K 是可表达列表依赖的架构，不等于已经优化列表效用。**

## 实验究竟证明了什么

论文只在 MovieLens-1M 上实验。表 3 的 NDCG@10 与 Recall@10：

![GPTRec 在 MovieLens-1M 上的结果](/blog/generative-recommendation/12-gptrec/evidence.svg)

*图 4：Top-K 接近最强基线；Next-K 付出明显准确率代价，换来列表条件建模能力。*

| 方法 | NDCG@10 | Recall@10 |
| --- | ---: | ---: |
| BERT4Rec | **0.152** | **0.282** |
| GPTRec-TopK | 0.146 | 0.254 |
| SASRec | 0.108 | 0.199 |
| GPTRec-NextK | 0.105 | 0.157 |

Next-K 的 NDCG 约为 Top-K 的 72%，论文概括为大约保留 75% 质量，同时与 SASRec 相当。这说明序列化列表并非免费：越靠后的错误会反过来污染后续条件。

多 token 实验展示了容量—质量折中：

| 表示 | embedding 数 | NDCG@10 | Recall@10 |
| --- | ---: | ---: | ---: |
| 一商品一 token | 3416 | 0.146 | 0.254 |
| 2 token × 2048 值 | 4096 | 0.124 | — |
| 4 token × 512 值 | **2048** | 0.108 | 0.182 |

四 token 表示比 3416 个 item embeddings 少约 40%，NDCG 与 SASRec 持平；两 token、较大桶数则取得更好精度。论文正文有一处把 one-token NDCG 写成 0.253 的疑似笔误，本篇采用表 3 的 0.146。

## 它失败在哪里

### 1. “能建模”不等于“学会多样”

Next-K 的因子分解允许后件依赖前件，但普通 next-item 训练仍主要奖励命中。没有覆盖率、多样性或长期满意度目标，就不能把结构能力当作业务效果。

### 2. 自回归错误会累积

第一个商品生成错误后，第二步把错误当事实；列表越长，暴露偏差越明显。训练时看真实前缀、推理时看自身前缀，也存在 teacher-forcing gap。

### 3. 延迟随 K 增长

Top-K 一次 scoring，Next-K 需要 K 次。多 token ID 又把每件商品展开成多步，在线成本可能乘起来。

### 4. SVD ID 缺少内容冷启动

它来自交互矩阵，新商品没有列信号；与 TIGER 的内容 SID 相反。SVD 更新后坐标和桶边界也可能变化，引出地址版本迁移。

### 5. 证据范围很窄

只有 MovieLens-1M，商品 3416 个，远不能证明多 token softmax 与 Next-K 在工业目录上的可扩展性。

## 下一篇为什么会出现

现在有两种商品“语言”：

- TIGER：从文本内容构造 Semantic ID，擅长语义与冷启动；
- GPTRec-SVD：从用户—商品矩阵构造协同 token，擅长行为结构。

真正的推荐需要两者。文本相似的商品不一定被同一群人喜欢；协同相似的商品也可能名字完全不同。[下一篇 LC-Rec](/blog/generative-recommendation-13-lc-rec-zh) 会把预训练 LLM、Semantic ID 和协同行为放进同一套对齐任务，并用均匀语义映射解决 ID 冲突。

---

**本篇批判性结论**：GPTRec 的意义在于拆开两个常被混在一起的问题——商品如何 token 化、列表如何因子化。它证明紧凑 token 与逐件生成都可行，也用准确率和延迟提醒我们：生成式接口只是扩大了可优化空间，列表目标本身仍要被明确写进训练信号。
