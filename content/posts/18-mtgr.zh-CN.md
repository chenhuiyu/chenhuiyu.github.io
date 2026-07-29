---
title: "传统交叉特征真的应该全部扔掉吗？｜生成式推荐 18：MTGR"
date: "2026-07-29"
updated: "2026-07-29"
category: "Generative Recommendation"
language: "zh-CN"
tags:
  - Generative Recommendation
  - Recommender Systems
  - MTGR
  - Industrial Ranking
pairKey: "generative-recommendation-18-mtgr"
slug: "generative-recommendation-18-mtgr-zh"
excerpt: "MTGR 保留 DLRM 的候选交叉特征，把同一用户的多候选聚合成一个 HSTU 序列，并用 GLN 与动态遮罩兼顾规模化、效果与因果安全。"
series: "generative-recommendation"
seriesOrder: 18
draft: false
---

生成式推荐有一个诱人的承诺：

> 只要保留足够长的原始行为、把模型做得足够大，手工统计和交叉特征最终都可以被模型重新学出来。

[HSTU](/blog/generative-recommendation-16-hstu-zh) 明确提出过这种方向，[OneRec](/blog/generative-recommendation-17-onerec-zh) 进一步尝试用一个生成器取代多级漏斗。

美团的实践给出了一盆很有价值的冷水。[MTGR: Industrial-Scale Generative Recommendation Framework in Meituan](https://arxiv.org/abs/2505.18654) 报告：

> 去掉经过长期验证的 candidate-aware cross features，模型性能显著下降；扩大纯生成模型也补不回来。

MTGR 没有因此退回传统 DLRM。它选择混合：保留交叉特征，把用户和多个候选重新排成 token 序列，用 HSTU 一次编码，再在候选位置输出 CTR/CTCVR 分数。

## 30 秒看懂本文

1. **OneRec/HSTU 路线留下的问题**：长序列带来 scaling，但删除“此用户对当前候选的历史 CTR/曝光”等交叉统计，可能丢掉极高效的先验。
2. **MTGR 的答案**：把同一用户请求中的 K 个候选聚合成一个样本，用户特征只算一次；每个候选 token 携带自己的 cross features；用 Group-Layer Normalization 和 Dynamic Masking 处理异构语义与时间泄露。
3. **最重要的证据**：去掉 cross features 后 CTCVR GAUC 从 0.6603 降到 0.6514，低于强 DLRM 的 0.6550；MTGR-large 达到 0.6646，线上 PV_CTR +1.90%、UV_CTCVR +1.02%。

![纯 GRM 与 MTGR](/blog/generative-recommendation/18-mtgr/paradigm-shift.svg)

*图 1：MTGR 的立场不是“生成或传统二选一”，而是把传统有效信息装进可复用的序列计算。*

## 什么是 cross feature

原始特征可能是：

```text
用户：城市、年龄、会员等级
历史：点击过哪些餐厅、最近买过什么
候选：商家、菜品、价格、配送距离
```

交叉特征会直接计算用户与候选之间的关系，例如：

```text
该用户过去对“此候选类目”的 CTR
该用户看过“此商家”的次数
用户常购价格区间与“此候选价格”的差
历史序列中与“此候选标签”匹配的行为数
```

它们确实是人工设计，却把大量事件压缩为一个低成本、target-aware 统计。在有限模型和有限上下文中，“让模型从一千条事件重新数一遍”未必比直接给计数更聪明。

问题是传统 DLRM 为每个候选重复用户计算：

```text
(同一用户历史, 候选1) → score1
(同一用户历史, 候选2) → score2
...
```

候选数增加，用户侧大网络也跟着重复 K 次。

## 用八件商品理解用户级压缩

把输入分为五组：

- \(U\)：用户静态特征；
- \(S\)：较长期行为序列；
- \(R\)：请求前几小时/一天内的实时行为；
- \(I_i\)：第 \(i\) 个候选自身特征；
- \(C_i\)：用户与第 \(i\) 个候选的 cross features。

传统数据有 K 行，MTGR 聚合为一行：

\[
D=
[U,S,R,[C,I]_1,\ldots,[C,I]_K].
\]

在八件商品例子中：

```text
共享一次：
  U = 用户城市/年龄
  S = I01 网球拍, I03 攀岩鞋, I05 羽毛球拍
  R = 刚刚收藏 I07 泳帽

候选 token 1：
  I06 泳镜 + 用户×游泳类 CTR + 历史曝光

候选 token 2：
  I08 运动毛巾 + 用户×恢复类 CTR + 历史曝光
```

一次 HSTU 前向输出 `score(I06), score(I08), ...`。用户表示被 K 个候选共享，推理成本对候选数呈次线性增长。

![用户级聚合](/blog/generative-recommendation/18-mtgr/toy-compression.svg)

*图 2：压缩的不是用户信息，而是“同一用户被重复计算”的样本冗余。每个候选仍保留自己独有的交叉统计。*

## 不同类型特征怎样变成 token

标量特征如年龄、性别各自映射成 token。序列中每件商品的多个字段先 embedding + concat，再经专属 MLP 投影到统一的 \(d_{\rm model}\)。

候选也一样，但其输入是 \([C_i,I_i]\)：cross feature 与 item feature 在候选 token 形成前就融合。最后：

\[
\operatorname{Feat}_D
=\operatorname{Concat}
([\operatorname{Feat}_U,\operatorname{Feat}_S,
\operatorname{Feat}_R,\operatorname{Feat}_I])
\in\mathbb R^{(N_U+N_S+N_R+N_I)\times d_{\rm model}}.
\]

候选位置的输出 hidden state 经 MLP 产生 CTR 与 CTCVR logits。这里没有自回归地生成商品 ID；“Generative”主要指用户级序列数据编排、长历史端到端学习和计算复用，任务末端仍是判别排序。

## 为什么需要 Group-Layer Normalization

年龄 token、历史商品 token、实时行为 token、候选交叉 token 来自完全不同分布。若所有 token 共用一套 LayerNorm 统计，一个 domain 的极值可能影响另一 domain。

MTGR 先按语义域分别归一化：

\[
\tilde X=\operatorname{GroupLN}(X),
\]

再投影为 \(Q,K,V,U\)，进入类似 HSTU 的 SiLU attention 与门控。GLN 的作用不是让语义相同，而是先把数值尺度对齐，让注意力不必同时学习“分布校准”和“信息交互”。

![MTGR 的 token 编码](/blog/generative-recommendation/18-mtgr/mechanism.svg)

*图 3：GroupLN 解决异构分布，Dynamic Mask 解决可见性，HSTU 负责交互；三者不能互相替代。*

## Dynamic Masking 为什么是必要条件

把同一用户几小时内多个训练事件聚合后，序列里会同时出现“过去”和“对较早 target 来说属于未来”的实时行为。若 full attention 全部可见，早期候选能偷看后续点击，产生标签泄露。

MTGR 按 token 类型设置可见性：

1. **静态序列 \(U,S\)**：发生在聚合窗口前，对所有位置可见；
2. **动态实时序列 \(R\)**：按时间 causal，只能看见此前事件；
3. **候选 token**：根据它与实时事件的时间关系建立可见性，并限制不合理的候选间互看。

因此不能简单套标准 causal mask：静态用户信息无需被遮住，而多个候选/实时事件又有各自时间戳。

## 训练系统也属于方法的一部分

论文在 TorchRec 上做了多项工业改造：

- dynamic hash table 支持新用户/商品 embedding 实时插入与删除；
- embedding ID 去重和 table 自动合并减少 lookup；
- 按真实序列长度动态平衡各 GPU batch，缓解长尾负载；
- sequence packing 防止不同用户互相 attention；
- mixed precision 与 operator fusion。

相对原 TorchRec，训练吞吐提升 1.6–2.4 倍，并在 100+ GPU 上保持扩展。没有这些系统工作，65 倍单样本 FLOPs 不可能以可接受成本上线。

## 实验究竟证明了什么

10 天工业数据包含：

| 统计 | 数量 |
| --- | ---: |
| 训练用户 | 2.1 亿 |
| 商品 | 430 万 |
| 曝光 | 237.4 亿 |
| 点击 | 10.8 亿 |
| 购买 | 1.8 亿 |

强 DLRM `UserTower-SIM` 每样本 0.86 GFLOPs；MTGR-small / medium / large 分别为 5.47、18.59、55.76 GFLOPs，large 约为 65 倍。

![MTGR 的主结果与 cross ablation](/blog/generative-recommendation/18-mtgr/evidence.svg)

*图 4：纵轴截断是为了看清工业 GAUC 的小差异；去掉 cross features 不仅损失增益，还低于强 DLRM。*

| 模型 | CTR GAUC | CTCVR GAUC |
| --- | ---: | ---: |
| UserTower-SIM | 0.6792 | 0.6550 |
| MTGR-small | 0.6826 | 0.6603 |
| MTGR-medium | 0.6843 | 0.6625 |
| MTGR-large | **0.6865** | **0.6646** |
| MTGR-small w/o cross | 0.6689 | 0.6514 |

去掉 GLN 后 CTCVR GAUC 为 0.6585，去掉 dynamic mask 为 0.6587，说明二者的贡献约与 small → medium 扩模相当。更关键的是，`w/o cross` 的 0.6514 比强 DLRM 0.6550 还低：

> 在这个场景里，架构 scaling 与有效特征不是替代关系。

线上用 2% 流量、六个月数据，对比持续优化两年的生产 DLRM：

| 模型 | PV_CTR | UV_CTCVR |
| --- | ---: | ---: |
| MTGR-small | +1.04% | +0.04% |
| MTGR-medium | +2.29% | +0.62% |
| MTGR-large | **+1.90%** | **+1.02%** |

medium 的 CTR 高于 large、large 的转化更高，说明多任务 scaling 并非每个线上指标单调同步。论文称最终模型已承接主流量，训练成本与 DLRM 相同，推理成本反而降低 12%。

## 它失败在哪里

### 1. 名字中的“Generative”容易误导

MTGR 不生成商品 ID 或列表，最终是候选位置的 discriminative logits。它继承的是 GR 的序列编排和计算复用，不应与 TIGER/OneRec 的生成式检索混为同一种输出范式。

### 2. 特征工程债务仍在

保留 cross features 也保留了维护成本：定义、离线/在线一致性、实时计算、漂移监控和潜在偏差都必须长期治理。

### 3. Dynamic mask 一旦出错就会泄露

多种 token、时间窗口和候选可见性让 mask 比标准 causal attention 复杂。离线提高若来自未来行为泄露，线上会消失；系统必须用严格的时间回放测试验证。

### 4. “scale 补不回来”是场景结论

证据来自美团外卖 ranking 和特定 cross features，不能推出所有平台都必须保留同样手工统计。更高质量的原始事件、更长上下文或不同自监督任务可能改变边界。

### 5. 大模型仍不等于长到无限

论文最长长期序列设为 1000，实时序列 100；这比许多 DLRM 更强，却仍是截断后的历史。65 倍 FLOPs 的收益也依赖专用训练/服务优化。

## 下一篇为什么会出现

HSTU、OneRec、MTGR 已经回答了规模、漏斗、特征与成本，但它们的决策仍主要是隐式 hidden state：

```text
模型给出 I06，
却不能自然处理“我今天肩膀受伤，别推荐高强度运动”这样的即时语言约束，
也不能展示它从哪些行为抽出了什么兴趣。
```

LLM 的另一项能力——在答案前生成文本推理——开始进入工业生成式推荐。[下一篇 OneRec-Think](/blog/generative-recommendation-19-onerec-think-zh) 会先把 itemic tokens 与文本语义对齐，再用推荐 CoT、GRPO 和 Think-Ahead 架构尝试“先想后推荐”。

---

**本篇批判性结论**：MTGR 是对技术潮流很健康的一次校准。Scaling 的正确问题不是“旧特征什么时候全部消失”，而是“哪些信息让大模型从原始数据重学最划算，哪些信息作为显式统计更高效”。工业最优解往往是重新组合，而不是纯粹替代。
