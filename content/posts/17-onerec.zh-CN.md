---
title: "召回和排序，能否由一个模型一次完成？｜生成式推荐 17：OneRec"
date: "2026-07-29"
updated: "2026-07-29"
category: "Generative Recommendation"
language: "zh-CN"
tags:
  - Generative Recommendation
  - Recommender Systems
  - OneRec
  - Preference Alignment
pairKey: "generative-recommendation-17-onerec"
slug: "generative-recommendation-17-onerec-zh"
excerpt: "OneRec 用 Balanced Semantic ID、Session-wise Generation、Sparse MoE 与迭代 DPO，把快手的召回—粗排—精排漏斗改写为一次列表生成。"
series: "generative-recommendation"
seriesOrder: 17
draft: false
---

传统推荐漏斗像接力赛：

```text
千万目录
  ↓ recall 留几千
  ↓ pre-rank 留几百
  ↓ rank 排最终几十
```

每一级都省了计算，也设置了下一棒的上限。召回漏掉的商品，精排再聪明也救不回来；各级分别优化点击、观看与互动，最终列表还需要规则拼接。

[HSTU](/blog/generative-recommendation-16-hstu-zh) 已把 ranking 与 retrieval 写进一个 sequential transduction 框架，却仍是不同位置、不同输出目标。[OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment](https://arxiv.org/abs/2502.18965) 提出更激进的工业方案：

> 不再逐级筛候选，而让一个模型根据用户历史直接生成一整个高价值 session。

## 30 秒看懂本文

1. **HSTU 留下的问题**：共享骨干不等于共享最终决策；多阶段漏斗仍有召回上限和目标割裂。
2. **OneRec 的答案**：用 Balanced residual K-means 造三级 Semantic ID，T5 encoder–decoder 生成 5–10 个视频 session，decoder 用 Sparse MoE 扩容量，再以 reward model 从自身 beam 候选中选 preference pairs 做迭代 DPO。
3. **最重要的证据**：离线最大 session watch-time 从 TIGER-1B 的 0.1368、OneRec-1B 的 0.1529 升到 IPA 的 0.1933；线上相对原多阶段系统总观看时长 +1.68%，单视频平均观看时长 +6.56%。

![从推荐漏斗到一次 session 生成](/blog/generative-recommendation/17-onerec/paradigm-shift.svg)

*图 1：所谓“单阶段”是一个模型直接产生可展示列表，但系统外围仍有 tokenizer、beam、奖励模型和在线更新。*

## 为什么 next-item 还不够

TIGER 式训练用：

```text
历史 → 下一件商品
```

线上需要 10 个视频时，系统可以独立生成 10 次，再用去重、多样性、创作者打散等规则拼起来。问题在于模型从没见过“十件作为整体”的目标：

- 第一条和第二条是否重复？
- 前三条是否都过于沉重？
- 应在什么位置插入新兴趣？
- 一段 session 的总观看价值如何权衡？

OneRec 把一次请求实际返回的一批短视频定义为 session，通常 5–10 个。高价值训练 session 需要满足：

1. 用户实际观看至少 5 个；
2. 总观看时长超过阈值；
3. 或发生点赞、收藏、分享等互动。

于是模型学习的是经过日志规则筛选的“好列表”，不是任意曝光列表。

## 用八件商品理解 session-wise generation

用户历史：

```text
I01 网球拍 → I03 攀岩鞋 → I05 羽毛球拍 → I07 泳帽
```

目标不再只是 `I06 泳镜`，而是一段：

```text
I06 泳镜 → I08 运动毛巾 → I04 镁粉袋 → I02 网球
```

生成 I08 时，decoder 已经看见 I06；生成 I04 时，也看见前两件。列表的内容和顺序都进入联合概率：

\[
P(S\mid H_u)
=\prod_{i=1}^{m}\prod_{j=1}^{L}
P(s_{i,j}\mid H_u,S_{<i},s_{i,<j}).
\]

- \(m\)：session 中 5–10 件商品；
- \(L\)：每件商品的 Semantic ID 层数；
- \(S_{<i}\)：已经生成的商品；
- \(s_{i,<j}\)：当前商品已经生成的 code 前缀。

![八件商品如何组成 session](/blog/generative-recommendation/17-onerec/toy-session.svg)

*图 2：session-wise training 把列表依赖写进概率分解；它仍然需要高价值 session 的数据定义。*

## 先把商品地址分配均匀

OneRec 从多模态视频 embedding 出发，使用多级 residual K-means，而非 RQ-VAE。普通聚类会产生 hourglass：某些 code 挤满视频，某些几乎为空，beam 在早期就被热门分支垄断。

Balanced K-means 令每个簇容量固定：

\[
w=\frac{|V|}{K}.
\]

对每个 centroid，按距离从尚未分配的视频中取最近的 \(w\) 个，再更新中心，循环到 assignment 收敛。每一级都对上一层残差继续量化：

\[
s_i^l=\arg\min_k\lVert r_i^l-c_k^l\rVert_2^2,
\qquad
r_i^{l+1}=r_i^l-c_{s_i^l}^l.
\]

距离保留相似性，固定容量保证 code 使用率。与 LC-Rec 的 Sinkhorn 软分配不同，这里是顺序、硬容量的 balanced K-means 算法。

## Sparse MoE 怎样扩大 OneRec

encoder 用 Transformer 编码正向观看、点赞、关注、分享等历史。decoder 逐 token 生成 session。

若把所有 FFN 都做大，每个 token 的 FLOPs 也线性增长。OneRec 把 decoder FFN 替换为 Sparse Mixture-of-Experts：

\[
h_{t}^{l+1}
=h_t^l+\sum_{i=1}^{N_{\rm MoE}}g_{i,t}\operatorname{FFN}_i(h_t^l),
\]

只有 router top-\(K_{\rm MoE}\) 的 \(g_{i,t}\) 非零。工业配置有 24 个 experts，每个 token 激活 2 个，因此模型可达 1B 参数，推理只使用约 13% 参数。

![OneRec 完整训练闭环](/blog/generative-recommendation/17-onerec/mechanism.svg)

*图 3：造码、列表 NTP 与偏好对齐是三层不同优化；Sparse MoE 解决容量，不负责定义“好列表”。*

## 为什么推荐版 DPO 不能直接照搬

语言 DPO 常有人类明确比较两个回答。推荐日志中，同一次请求只展示一张列表；用户不可能同时观看另一张反事实列表，因此没有天然的 chosen/rejected pair。

OneRec 先训练 session reward model \(R(u,S)\)。对每件视频构造 user-target interaction representation，session 内再 self-attention，分别预测：

- `swt`：session watch time；
- `vtr`：view probability；
- `wtr`：follow probability；
- `ltr`：like probability。

它们用真实日志的多任务 BCE 训练。随后 Iterative Preference Alignment（IPA）循环：

```text
当前 OneRec 对用户做 beam search，生成 N=128 个 sessions
reward model 给 128 个结果打分
最高分 → chosen S_w
最低分 → rejected S_l
以 NTP + λDPO 更新模型
保存新快照，继续挖自己的 hard negatives
```

DPO 项为：

\[
\mathcal L_{\rm DPO}
=-\log\sigma\left[
\beta\log\frac{\pi_\theta(S_w\mid H)}{\pi_{\rm ref}(S_w\mid H)}
-\beta\log\frac{\pi_\theta(S_l\mid H)}{\pi_{\rm ref}(S_l\mid H)}
\right].
\]

由于生成 128 个候选很贵，实验只有 \(r_{\rm DPO}=1\%\) 的样本做 IPA。把比例增到 5% 需要约 5 倍 GPU，但收益很小；1% 已获得平均约 95% 的最大观察效果。

## 实验究竟证明了什么

离线指标不是传统 Recall/NDCG，而是 reward model 对生成 session 的预测均值与最大值。关键最大值如下：

![TIGER、OneRec 与 IPA](/blog/generative-recommendation/17-onerec/evidence.svg)

*图 4：session-wise generation 本身高于 point-wise TIGER，IPA 又显著拉高 reward model 认为最有价值的候选。*

| 模型 | max swt | max ltr |
| --- | ---: | ---: |
| TIGER-1B | 0.1368 | 0.0579 |
| OneRec-1B | 0.1529 | 0.0660 |
| OneRec-1B + IPA | **0.1933** | **0.1203** |

IPA 相对基础 OneRec-1B，max swt +4.04%、max ltr +5.43% 是论文按自己的比较口径报告的相对变化；表中绝对数值差看起来更大，是因为还涉及不同统计切片与重复实验汇总，读者应以表格定义为准，不把二者混算。

模型从 0.05B 扩到 0.1B，accuracy 最大提升 14.45%；继续到 0.2B、0.5B、1B 又分别有 5.09%、5.70%、5.69% 额外增益。

线上以 1% 主流量对比现有多阶段系统：

| 模型 | Total Watch Time | Average View Duration |
| --- | ---: | ---: |
| OneRec-0.1B | +0.57% | +4.26% |
| OneRec-1B | +1.21% | +5.01% |
| OneRec-1B + IPA | **+1.68%** | **+6.56%** |

这比离线 reward 指标更有说服力：至少在快手短视频主场景中，一个生成 session 的单模型确实超过了成熟级联基线。

## 它失败在哪里

### 1. “偏好”是 reward model 的偏好

chosen 和 rejected 都未真实展示给用户，而是奖励模型在 OneRec 自己的候选中挑选。RM 有偏差时，DPO 会放大偏差；模型还可能学会 exploit RM，而非真正改善长期满意度。

### 2. 高价值 session 带选择偏差

只保留观看多、时长高、有互动的列表，会偏向活跃用户、热门内容与已有策略。未观看可能是内容差，也可能是位置靠后或用户临时离开。

### 3. 单阶段不是单组件

系统仍需要多模态 encoder、balanced codebook、beam search、24-expert MoE、reward model、DPO sample server、在线增量训练与 code 映射。它消除的是 ranker 级联，不是所有系统复杂度。

### 4. 列表依赖会积累错误

早期 Semantic ID 出错会成为后续上下文。beam search 缓解但增加延迟；平衡 code 让树好走，却可能切开自然的内容簇。

### 5. 外部验证很有限

核心数据与指标私有，arXiv 版本还保留了未清理的会议模板占位。线上结果重要，但难由外部团队复现或判断实验长期稳定性。

## 下一篇为什么会出现

OneRec 的叙事是“生成模型取代级联”。但 HSTU/纯 GR 的另一项主张是：原始序列足够长、模型足够大，就可以放弃传统手工统计和 cross features。

美团团队观察到相反现象：

> 去掉 candidate-aware cross features 会造成显著质量损失，扩大生成模型也补不回来。

[下一篇 MTGR](/blog/generative-recommendation-18-mtgr-zh) 不会否定生成式规模化，而是提出一个更务实的混合答案：保留 DLRM 的有效特征，把同一用户的多个候选聚合成 token 序列，用 HSTU 一次编码并输出多个判别分数。

---

**本篇批判性结论**：OneRec 首次把“最终展示 session”变成工业生成目标，并用线上实验说明统一漏斗并非空想；但它没有消灭推荐系统的中间层，只是把复杂度从多个 ranker 转移到了商品语言、生成解码和偏好对齐。
