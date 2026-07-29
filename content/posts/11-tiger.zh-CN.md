---
title: "给每件商品一个语义地址｜生成式推荐 11：TIGER"
date: "2026-07-29"
updated: "2026-07-29"
category: "Generative Recommendation"
language: "zh-CN"
tags:
  - Generative Recommendation
  - Recommender Systems
  - TIGER
  - Semantic ID
pairKey: "generative-recommendation-11-tiger"
slug: "generative-recommendation-11-tiger-zh"
excerpt: "TIGER 用 RQ-VAE 把商品内容压缩成粗到细的 Semantic ID，再让 T5 逐 token 生成下一件商品，为生成式推荐建立了经典范式。"
series: "generative-recommendation"
seriesOrder: 11
draft: false
---

[上一篇 DSI](/blog/generative-recommendation-10-dsi-zh) 让模型直接生成文档 ID，但它的 semantic string 由分层 k-means 构造，研究对象还是文档检索。

[TIGER: Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) 把这条路线正式带进序列推荐。它不再给每件商品一个巨大、互不相关的分类 token，而是把商品内容压缩成一串粗到细的离散码：

```text
泳镜 I06 → 〈水上运动, 游泳, 眼部装备, 唯一后缀〉
```

模型要做的，不是从数万商品中一次选中 I06，而是一步步把它的“语义地址”写出来。

## 30 秒看懂本文

1. **DSI 留下的问题**：任意 ID 没有语义，atomic ID 的输出层随目录增长；分层聚类又与生成模型分离。
2. **TIGER 的答案**：先用 Sentence-T5 得到商品内容向量，再用三层 residual quantization 学 Semantic ID，最后追加碰撞 token 保证唯一；T5-like 模型根据用户历史生成下一个 ID。
3. **最重要的证据**：在 Beauty 上，TIGER 的 NDCG@5 为 0.0321，SASRec 为 0.0249，提升 29.04%；把 RQ tokenizer 换成 Random ID 后只剩 0.0205。

![从商品分类到语义地址生成](/blog/generative-recommendation/11-tiger/paradigm-shift.svg)

*图 1：多 token 地址把一个超大的平面分类问题，改写成共享前缀的层次决策。*

## DSI 的地址为什么还不够适合商品

商品目录有三个特点。

第一，商品不断新增。每个商品一个 atomic token，会让 embedding 表和 softmax 随目录线性增长。

第二，新商品没有点击，却有标题、品牌、价格、类目等内容。如果 ID 能从内容计算，新商品无需重新训练整套推荐器就能进入地址空间。

第三，商品之间不是孤立类别。泳镜和泳帽应该先共享“水上—游泳”前缀，再在末端分开。共享前缀能让稀疏商品复用统计强度。

TIGER 因此把系统拆成两个阶段：

```text
内容 → Semantic ID tokenizer
行为历史中的 Semantic IDs → generative recommender
```

这套“先造词，再学句子”的架构，成为后续 Semantic ID 推荐的标准蓝图。

## 用八件商品理解 Residual Quantization

假设 Sentence-T5 已把八件商品映射到 768 维内容向量。一个普通 VQ-VAE 会用最近的 code 表示整个向量，但单码本要覆盖所有细节，容易需要大量 code。

RQ-VAE 改成逐层解释残差。

以 `I06 泳镜` 为例：

1. 第一层选择 code 17，解释“运动”；
2. 从原向量减去 code 17，第二层在剩余信息中选择 code 83，解释“水上/游泳”；
3. 再减残差，第三层选择 code 41，解释“眼部装备/防雾”；
4. 如果另一副泳镜得到相同前三码，再追加 collision token 区分实例。

于是：

```text
I06 泳镜 = 〈17, 83, 41, 0〉
I07 泳帽 = 〈17, 83, 92, 0〉
```

前两码相同体现语义邻近，完整四码仍唯一。

![八件商品的 Semantic ID](/blog/generative-recommendation/11-tiger/toy-semantic-id.svg)

*图 2：共享前缀不是人工命名，而是内容向量经过残差量化后形成的粗到细结构。图中标签是便于理解的解释，不是模型真的看到的词。*

## 核心公式：每一层只编码尚未解释的部分

令初始残差为内容 latent \(r_1=z\)，第 \(l\) 层从码本 \(\{e_k^l\}_{k=1}^K\) 中选最近 code：

\[
c_l=\arg\min_k\lVert r_l-e_k^l\rVert_2^2,
\qquad
r_{l+1}=r_l-e_{c_l}^l.
\]

三层之后，重构向量为：

\[
\hat z=e_{c_1}^1+e_{c_2}^2+e_{c_3}^3.
\]

- \(c_1\) 解释最大尺度结构，例如运动类别；
- \(r_2\) 是第一层没解释的细节；
- \(c_2,c_3\) 逐步细化；
- \(\hat z\) 应接近原 latent \(z\)。

RQ-VAE 的训练还包含重构损失与 commitment/codebook 项。关键是：组合数约为 \(K^L\)，参数量却只与 \(K\times L\) 成正比。

论文使用 3 个量化层、每层 256 个 code、32 维 latent；另加第 4 个碰撞位置，因此生成词表只需 \(256\times4=1024\) 个 code token，而不是为每件商品准备独立词元。

## 模型从输入到输出发生了什么

用户历史先替换成 Semantic ID：

```text
I01 → 〈12, 07, 44, 0〉
I03 → 〈31, 18, 09, 0〉
I05 → 〈12, 56, 22, 0〉
I07 → 〈17, 83, 92, 0〉
```

这些 token 连同 hashed user token 进入 T5-like encoder。decoder 自回归生成下一件商品的四个 code。论文模型约 13M 参数，用户 ID 被哈希到 2000 个 user tokens，最大历史长度为 20。

![TIGER 的两阶段机制](/blog/generative-recommendation/11-tiger/mechanism.svg)

*图 3：tokenizer 学“商品怎样命名”，生成器学“用户下一步会说出哪个名字”。两者在原始 TIGER 中分开训练。*

## 训练和推理怎么做

### 阶段一：训练 tokenizer

输入字段包含标题、价格、品牌、类目等，经 Sentence-T5 得到 768 维语义向量；RQ-VAE 压缩到 32 维 latent 并学习三级 codebook。

### 阶段二：训练推荐器

对每条序列构造 next-item 样本：

```text
〈SID(I01), SID(I03), SID(I05)〉 → SID(I07)
```

目标仍是逐 token 交叉熵：

\[
\mathcal L_{\text{rec}}
=-\sum_t\log P_\theta(c_t^{\,*}\mid c_{<t}^{\,*},H_u).
\]

### 推理

beam search 生成多个四-token 地址，再映射回商品。论文报告 top-10 中非法 ID 比例约 0.1%–1.6%；可扩大 beam 后过滤。今天更常见的实现会直接用目录 trie 约束合法路径。

这种推理比 ANN 慢：每个 code 都要一次 decoder 前向，beam 还要保留多个分支。论文明确说明效率不是其优化目标。

## 实验究竟证明了什么

三个 Amazon 子集只有约 10K–20K 商品，但结果清楚地区分了“会生成”与“ID 设计得好”。

主实验中：

| 数据集 | 强基线 NDCG@5 | TIGER NDCG@5 | 相对变化 |
| --- | ---: | ---: | ---: |
| Sports | S3-Rec 0.0161 | **0.0181** | +12.42% |
| Beauty | SASRec 0.0249 | **0.0321** | +29.04% |
| Toys | SASRec 0.0306 | **0.0371** | +21.24% |

更有解释力的是 tokenizer 消融：

![不同 ID 设计的效果](/blog/generative-recommendation/11-tiger/evidence.svg)

*图 4：同一个生成框架下，RQ Semantic ID 稳定高于 Random ID 与 LSH，说明地址结构本身就是模型能力的一部分。*

| Tokenizer | Sports | Beauty | Toys |
| --- | ---: | ---: | ---: |
| Random ID | 0.0050 | 0.0205 | 0.0270 |
| LSH | 0.0146 | 0.0259 | 0.0299 |
| RQ Semantic ID | **0.0181** | **0.0321** | **0.0371** |

冷启动实验则移除 5% 测试商品的训练交互。新商品仍可由内容得到 SID；检索时把与候选共享前三码的新商品纳入结果。这证明 Semantic ID 提供了一条内容冷启动通道，不过该规则带有人工的 \(\epsilon\) 上限，并非模型自然学出的完整冷启动策略。

还有一个值得记录的数据处理细节：论文指出此前 P5 的公开处理方式先把全数据中的商品映射为连续 ID 再切分，可能泄露测试目录信息；TIGER 重新处理以避免该问题。生成式方法尤其容易从“词表里出现过某商品”获得隐性目录信息。

## 它失败在哪里

### 1. tokenizer 只看内容，不看协同行为

文本相似不代表同一批用户喜欢。跑鞋与登山鞋语义接近，消费人群和下一步行为却可能不同。TIGER 的地址没有显式吸收共现结构。

### 2. 两阶段训练会失配

RQ-VAE 优化重构，推荐器优化 next-item。对内容重构最好的 code，不一定最容易预测，也不一定最适合排序；推荐损失无法反向修正 tokenizer。

### 3. 唯一性靠碰撞 token 补丁

前三层 code 冲突时，用第四层任意后缀区分。它保证地址合法，却没有给最后一位语义。

### 4. 规模与延迟尚未证明

目录只有万级，历史最多 20。自回归 beam search 比一次 ANN 查询昂贵得多，论文没有工业吞吐实验。

### 5. 地址存在版本问题

只要重训 tokenizer，某件商品的 SID 就可能改变。历史日志、缓存、在线 trie 与生成模型必须同时迁移，否则同一串 code 会指向不同商品。

## 下一篇为什么会出现

TIGER 把“下一个商品”写成了一个 ID 序列，但推荐最终通常要返回一个列表。若每个商品都独立打分，列表成员之间无法直接相互影响；若逐件生成，第二件就可以看到第一件，从而表达多样性、覆盖与顺序。

[下一篇 GPTRec](/blog/generative-recommendation-12-gptrec-zh) 同时探索两件事：用 SVD 把商品拆成多个紧凑 token，以及把 Top-K 改写为 Next-K，让模型一件一件写出推荐列表。

---

**本篇批判性结论**：TIGER 的核心贡献不是“用了 T5”，而是建立了生成式推荐的三件套：可组合的商品词表、序列到地址的生成目标、地址到目录的解码接口。它也让研究重心从“推荐模型多大”转向“商品应该怎样被 token 化”。
