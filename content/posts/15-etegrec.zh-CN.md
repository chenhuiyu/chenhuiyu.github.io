---
title: "Tokenizer 不该只是离线预处理｜生成式推荐 15：ETEGRec"
date: "2026-07-29"
updated: "2026-07-29"
category: "Generative Recommendation"
language: "zh-CN"
tags:
  - Generative Recommendation
  - Recommender Systems
  - ETEGRec
  - End-to-End Tokenization
pairKey: "generative-recommendation-15-etegrec"
slug: "generative-recommendation-15-etegrec-zh"
excerpt: "ETEGRec 用序列—商品与偏好—语义两类对齐损失连接 tokenizer 和生成器，再通过交替优化让商品 ID 与用户表示共同演化。"
series: "generative-recommendation"
seriesOrder: 15
draft: false
---

过去四篇的共同流程是：

```text
先训练 tokenizer → 冻结商品 ID → 再训练推荐器
```

这像先由词典编纂者造出一套词，再让作家必须用它写小说。若某个词难以从上下文预测，作家不能要求词典修改；若作家的语境发生变化，词典也看不见。

[LETTER](/blog/generative-recommendation-14-letter-zh) 已让 tokenizer 参考传统协同 embedding，但最终生成器仍只能接受既定地址。[ETEGRec: End-to-End Learnable Item Tokenization for Generative Recommendation](https://arxiv.org/abs/2409.05546) 进一步让两个模型互相对齐：

- 历史 encoder 应与目标商品预测出相同的 code 分布；
- decoder 的用户偏好状态应靠近目标商品的重构语义；
- tokenizer 与 recommender 交替更新，逐轮共同演化。

## 30 秒看懂本文

1. **LETTER 留下的问题**：tokenizer 虽然吸收协同信号，却不知道最终生成器如何表示历史与偏好；一旦冻结，推荐损失不能改变商品 code。
2. **ETEGRec 的答案**：用 SIA 在离散 code 分布层面对齐序列与目标商品，用 PSA 在连续表示层面对齐 decoder 偏好与目标商品；每个 cycle 先更新 tokenizer，再更新 recommender。
3. **最重要的证据**：Game 上 ETEGRec 的 NDCG@10 为 0.0507，LETTER 为 0.0475；去掉 alternating training 后降到 0.0428，降幅大于去掉任一单独对齐项。

先给“端到端”加一个精确限定：

> ETEGRec 不是让梯度一次穿过离散 argmin，从生成 loss 直接回到 codebook；它用共享对齐损失和冻结—切换的交替训练实现 co-evolution。更准确地说，是目标耦合的端到端，而不是单图全可微。

![从冻结词典到共同演化](/blog/generative-recommendation/15-etegrec/paradigm-shift.svg)

*图 1：两边通过中间表示互相提供监督，但每个更新阶段仍会冻结另一边以保持稳定。*

## 用八件商品理解两种“同意”

目标商品仍是 `I06 泳镜`，真实历史为：

```text
I01 网球拍 → I03 攀岩鞋 → I05 羽毛球拍 → I07 泳帽
```

tokenizer 看到 I06 的协同 embedding \(z\)，量化为 `〈4,2,1〉`，并重构出 \(\tilde z\)。

推荐器有两个不同视角：

1. encoder 读完历史后得到序列状态 \(z^E\)：它回答“接下来应该指向哪类商品 code？”
2. decoder 从 BOS 开始的首个状态 \(h^D\)：它概括当前用户偏好，回答“我正在为怎样的用户选商品？”

ETEGRec 要求两种一致：

```text
SIA：z^E 与目标商品 z 在每一级 code 上意见一致
PSA：h^D 与目标商品重构表示 z̃ 在连续空间靠近
```

前者对齐“地址”，后者对齐“含义”。

![八件商品中的双重对齐](/blog/generative-recommendation/15-etegrec/toy-coevolution.svg)

*图 2：SIA 连接 encoder 与量化器，PSA 连接 decoder 与商品重构表示，形成两条跨模型桥。*

## SIA：在每一级 code 分布上对齐

推荐器 encoder 的历史 hidden states 经 mean pooling 得到 \(z^E\)。商品 tokenizer 则接收目标商品的协同表示 \(z\)。把两者分别送入同一套 residual quantizer，在第 \(l\) 层得到 soft code 分布：

\[
p_l(k\mid z),\qquad p_l(k\mid z^E).
\]

Sequence–Item Alignment 最小化各层的对称 KL：

\[
\mathcal L_{\text{SIA}}
=\sum_l
\Big[
D_{\mathrm{KL}}(p_l(\cdot\mid z)\Vert p_l(\cdot\mid z^E))
+D_{\mathrm{KL}}(p_l(\cdot\mid z^E)\Vert p_l(\cdot\mid z))
\Big].
\]

论文公式排版在求和前出现负号，但 KL 作为对齐项的语义是最小化分布差异；这里按该意图表述。

若 tokenizer 认为 I06 第一层 code 4 概率 0.9，而历史状态认为 code 7 概率 0.8，SIA 会同时推动：

- tokenizer 的 code 边界更符合可预测历史；
- encoder 的序列表示更接近真实目标地址。

## PSA：让“用户想要什么”靠近“商品是什么”

Preference–Semantic Alignment 使用 decoder 的首个状态 \(h^D\) 和 tokenizer 重构的目标表示 \(\tilde z\)。在 batch 内，同一条样本是正例，其他目标商品是负例，做双向 InfoNCE：

\[
\mathcal L_{\text{PSA}}
=\frac12\left(
\ell_{h\to z}+\ell_{z\to h}
\right).
\]

一个方向可写成：

\[
\ell_{h\to z}
=-\sum_i\log
\frac{\exp(\operatorname{sim}(h_i^D,\tilde z_i)/\tau)}
{\sum_j\exp(\operatorname{sim}(h_i^D,\tilde z_j)/\tau)}.
\]

SIA 是离散层级的逐码一致，PSA 是连续空间的全局相似；两者不是重复损失。

## 模型从输入到输出发生了什么

实验先用 SASRec 学到 256 维 collaborative item embeddings，再以它们初始化 RQ-VAE。tokenizer 使用 3 层、每层 256 个 code，另加 collision token。生成器是 T5-like encoder–decoder。

![ETEGRec 的双对齐机制](/blog/generative-recommendation/15-etegrec/mechanism.svg)

*图 3：商品协同表示、encoder 序列状态和 decoder 偏好状态在两个层面汇合。*

两个子系统的目标分别为：

\[
\mathcal L_{\mathrm{IT}}
=\mathcal L_{\mathrm{SQ}}
+\mu\mathcal L_{\mathrm{SIA}}
+\lambda\mathcal L_{\mathrm{PSA}},
\]

\[
\mathcal L_{\mathrm{GR}}
=\mathcal L_{\mathrm{REC}}
+\mu\mathcal L_{\mathrm{SIA}}
+\lambda\mathcal L_{\mathrm{PSA}}.
\]

- \(\mathcal L_{\mathrm{SQ}}\)：tokenizer 的 semantic quantization；
- \(\mathcal L_{\mathrm{REC}}\)：生成器的 next-item token likelihood；
- 两个 alignment loss 同时出现在两边，成为公共语言。

## 为什么不能两边同时更新

商品 code 是离散且会跳变的。如果 tokenizer 每个 step 都改 codebook，同时生成器也在追逐新标签，目标会像移动靶：昨天的 `〈4,2,1〉` 今天可能表示另一件商品。

ETEGRec 用交替优化：

```text
repeat cycle:
    冻结 recommender
    用 L_IT 更新 tokenizer 1 个 epoch

    冻结 tokenizer
    用 L_GR 更新 recommender C−1 个 epoch

tokenizer 收敛后：
    永久冻结 tokenizer
    完成剩余 recommender 训练
```

它承认“共同演化”和“标签稳定”之间的冲突：早期允许 ID 改进，后期必须锁定词表。

## 实验究竟证明了什么

三个数据集约有 24.5K–25.8K 商品，采用 full-ranking，最大序列长度 50。

![ETEGRec、LETTER 与去掉交替训练的结果](/blog/generative-recommendation/15-etegrec/evidence.svg)

*图 4：完整模型稳定高于 LETTER；去掉交替训练反而低于 LETTER，说明耦合目标只有在稳定更新节奏下才有效。*

| 数据集 | LETTER | ETEGRec | w/o alternating |
| --- | ---: | ---: | ---: |
| Instrument | 0.0310 | **0.0331** | 0.0277 |
| Scientific | 0.0230 | **0.0241** | 0.0198 |
| Game | 0.0475 | **0.0507** | 0.0428 |

去掉两种 alignment 后，三个 NDCG@10 为 0.0317、0.0224、0.0478；仍高于“同时/非交替更新”变体。这支持两层结论：

1. SIA 与 PSA 提供增益；
2. **优化日程比增加损失本身更关键**。

论文报告完整模型在所有指标上相对最强基线具有 \(p<0.01\) 的显著性。但绝对提升并不巨大：例如 Game 是 0.0475 到 0.0507。它更像验证“tokenizer 应看见下游状态”的方向，而非说明工业问题已被解决。

## 它失败在哪里

### 1. “端到端”仍是交替近似

hard quantization 的 argmin 不可微，两个模块不是一次反向传播共同更新。系统需要维护冻结状态、cycle 长度和收敛判据。

### 2. 共同演化天然不稳定

论文自己的消融表明，不做交替更新会大幅退化。目录一旦在线变化，重新启动 co-evolution 可能再次移动全部地址。

### 3. 最终仍要冻结 tokenizer

模型先承认静态 ID 的缺陷，又在收敛后回到静态 ID，因为服务系统必须有稳定映射。这不是矛盾，而是现实约束尚未被解决的证据。

### 4. 训练与版本管理复杂

初始化 SASRec、训练 RQ-VAE、运行多 cycle、同步 ID 化历史、再完成生成器训练；任一步更新都要保证模型、codebook、trie 和缓存一致。

### 5. 仍未跨过工业规模

约 25K 商品、离线 full-ranking 不能回答千万目录的 beam latency、流式新增、删除、跨域与实时反馈问题。

## 下一篇为什么会出现

到这里，Semantic ID 路线完成了一个小闭环：

```text
DSI：ID 可以生成
TIGER：ID 可以由商品内容量化
LC-Rec：ID 可以连接 LLM 与协同任务
LETTER：tokenizer 可以显式优化协同与均衡
ETEGRec：tokenizer 可以看见下游推荐状态
```

但这些工作大多在万级目录、离线数据、encoder–decoder beam search 中验证。真正进入工业系统，还需要：

- 能处理超长历史和海量样本的高效架构；
- 把检索、排序、策略统一而不让延迟爆炸；
- 在生成过程中注入业务目标和推理反馈。

[下一篇 HSTU](/blog/generative-recommendation-16-hstu-zh) 将镜头从“怎样造商品 token”拉到“怎样训练一个工业级生成式推荐骨干”：重新设计注意力、时间与特征交互，让模型规模和用户历史一起扩展。

---

**本篇批判性结论**：ETEGRec 最重要的发现可能正是它的限制——商品词表若要适应下游，就必须变化；线上系统若要稳定，商品词表又必须冻结。交替优化是两者之间务实的桥，却还不是最终答案。
