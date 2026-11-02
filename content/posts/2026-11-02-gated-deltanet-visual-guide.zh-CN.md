---
title: "图解 Gated DeltaNet：固定大小的记忆，如何遗忘与更新"
date: "2026-11-02"
updated: "2026-11-02"
category: "NLP Insights"
language: "zh-CN"
tags: ["Paper Notes", "Gated DeltaNet", "Linear Attention", "LLM Architecture"]
pairKey: "gated-deltanet-visual-guide"
slug: "gated-deltanet-visual-guide-zh"
excerpt: "从 KV cache 的存储压力出发，用六张原创配图拆解 Gated DeltaNet 的状态更新、门控、分块训练与论文证据。"
draft: false
---

<img src="/blog/gated-deltanet/cover-zh.svg" width="1200" height="940" alt="Gated DeltaNet 视觉导读：衰减旧状态，按预测误差修正，再读出结果。">

读完 [KV cache 的续写机制](/blog/llm-infra-kv-cache-zh)，一个自然的问题是：如果不逐个保存历史 token 的 K 和 V，能否把它们写进一个大小固定的状态里？

这个选择把问题从“历史怎样存下来”推进到“有限的状态怎样更新”。旧信息要保留多久？同一个 key 的答案变了，怎样修正？无关内容不断写入，又会不会把重要关联淹没？

《Gated Delta Networks: Improving Mamba2 with Delta Rule》将两种控制结合在一条更新式里：**用门控衰减旧状态，用 delta rule 根据当前预测误差修正关联。** 本文沿着“存储 → 更新 → 训练 → 证据”的路线解释这件事。[1]

论文：Songlin Yang、Jan Kautz、Ali Hatamizadeh，ICLR 2025。解读固定采用 **arXiv v3，2025-03-06**。机制图与手算例子为本文原创；结果图重绘自原文表格。

## 1. 固定大小的状态，究竟省掉了什么

在普通多头注意力的某一层、某一个头里，若已读入 $T$ 个 token，key 和 value 维度分别是 $d_k,d_v$，历史 KV 需要保存 $T(d_k+d_v)$ 个元素。这里先忽略 batch、head 数、其他缓存与数据类型；GQA 等架构也需要另记账。

线性注意力提供了另一种组织方式。先看省略核特征映射和归一化项的基本形式：

$$
S_t=S_{t-1}+v_tk_t^\top,
\qquad o_t=S_tq_t,
\qquad S_t\in\mathbb{R}^{d_v\times d_k}.
$$

把递推展开，得到

$$
o_t=\sum_{i=1}^{t}v_i(k_i^\top q_t).
$$

这里每一项外积 $v_i k_i^\top$ 把一个 key–value 关联叠加进矩阵。查询 $q_t$ 与哪个历史 key 更接近，那个 value 对输出的贡献就更大。固定维度下，矩阵的元素数是 $d_vd_k$，不会随 $T$ 增长。[2]

上式中的点积系数可以为负，不能按 softmax 概率解读。这里已经改变了注意力规则；标准 softmax attention 无法直接按这条结合律整理成同一个递推。

<img src="/blog/gated-deltanet/storage-zh.svg" width="1200" height="930" alt="逐 token 保存的 KV 随历史长度增长；递归状态以一个固定大小的矩阵汇总历史。">

*图 1｜存储方式的变化。色块表示结构，不是实测激活。[查看原尺寸](/blog/gated-deltanet/storage-zh.svg)*

例如单头 $d_k=d_v=128$，历史长度 $T=8192$，历史 KV 是 $2{,}097{,}152$ 个元素，状态矩阵是 $16{,}384$ 个元素。如果二者都按两字节存储，对应约 4 MiB 和 32 KiB。**这只是上述简化对象的元素账本。** 实际递归状态可能保留 FP32，还存在短卷积缓存等开销，不能据此宣布整模型显存下降 128 倍。

更根本的变化是信息的组织形式：历史 token 不再逐个保留，而是共享一个有限维的关联空间。读取得到的是许多关联共同作用的结果。

## 2. 从累加，到衰减，再到按误差修正

为了保持符号一致，后文将 $q_t,k_t,v_t$ 都视为列向量，状态始终是 $d_v\times d_k$ ；示例中的 key 使用单位长度。

### 累加：新的关联总会写进去

基本线性注意力持续加上 $v_t k_t^\top$。若同一个单位 key 先后对应 $v_a$ 和 $v_b$，从零状态写入后，对这个 key 的读取会包含 $v_a+v_b$。它不会自动推断“后一个值应替换前一个值”。

### 衰减：让整个旧状态变淡

论文用下式概括 Mamba2 的相关递推结构，省略其完整参数化：[1][3]

$$
S_t=\alpha_tS_{t-1}+v_tk_t^\top,
\qquad 0<\alpha_t<1.
$$

$\alpha_t$ 小时，旧状态快速衰减。对一个头的这次更新而言，这个标量作用于整个状态矩阵。好处是容易清理历史，代价是有用的旧关联也会受到衰减。

### Delta rule：先问当前状态预测了什么

给定新来的 $k_t$，当前状态对其 value 的预测是 $S_{t-1}k_t$。DeltaNet 写入的是预测残差：[4]

$$
\begin{aligned}
e_t&=v_t-S_{t-1}k_t,\\
S_t&=S_{t-1}+\beta_t e_t k_t^\top.
\end{aligned}
$$

$\beta_t$ 控制写入强度。在单位 key 的前提下，直接右乘 $k_t$ 可得

$$
S_tk_t=(1-\beta_t)S_{t-1}k_t+\beta_tv_t.
$$

这是一种向新目标靠拢的插值。若允许取 $\beta_t=1$ 这个边界值，当前 key 的读取就会被替换为 $v_t$。论文主要采用的门值位于开区间，边界值帮助我们理解极限行为。

### Gated delta rule：先衰减，再计算残差

现在把两种控制结合起来。先定义衰减后的状态 $\bar S_t=\alpha_tS_{t-1}$：

$$
\begin{aligned}
e_t&=v_t-\bar S_tk_t,\\
S_t&=\bar S_t+\beta_t e_tk_t^\top\\
&=\alpha_tS_{t-1}(I-\beta_tk_tk_t^\top)
+\beta_tv_tk_t^\top.
\end{aligned}
$$

这就是原文式 (10)。**残差里的预测也必须来自已经衰减的状态。** 如果写成 $v_t-S_{t-1}k_t$，再在外面加一次衰减，就成了另一条更新规则。

<img src="/blog/gated-deltanet/rules-zh.svg" width="1200" height="1010" alt="四种更新规则：线性注意力累加；简化 Mamba2 全局衰减；DeltaNet 按残差校正；Gated DeltaNet 同时使用两种控制。">

*图 2｜把四种规则放在同一组符号里。色块为示意，Mamba2 只展示相关递推结构。[查看原尺寸](/blog/gated-deltanet/rules-zh.svg)*

## 3. 手算一次：两个门分别做了什么

取一个二维例子：

$$
S_{t-1}=\begin{bmatrix}1&0\\0&1\end{bmatrix},\quad
k_t=q_t=\begin{bmatrix}1\\0\end{bmatrix},\quad
v_t=\begin{bmatrix}0\\1\end{bmatrix},\quad
\alpha_t=0.8,\quad\beta_t=0.5.
$$

先衰减，得到 $\bar S_t=0.8I$。对当前 key 的预测是 $[0.8,0]^\top$，所以残差为 $[-0.8,1]^\top$。乘以写入强度，再与 key 做外积：

$$
\Delta S_t=
0.5\begin{bmatrix}-0.8\\1\end{bmatrix}
\begin{bmatrix}1&0\end{bmatrix}
=\begin{bmatrix}-0.4&0\\0.5&0\end{bmatrix}.
$$

最终

$$
S_t=\begin{bmatrix}0.4&0\\0.5&0.8\end{bmatrix},
\qquad o_t=S_tq_t=\begin{bmatrix}0.4\\0.5\end{bmatrix}.
$$

<img src="/blog/gated-deltanet/update-zh.svg" width="1200" height="1110" alt="二维手算：单位矩阵先乘 0.8，再加入仅第一列非零的残差外积，最终读出 0.4 和 0.5。">

*图 3｜衰减、校正、读出。数字完全来自给定公式，不涉及模型训练。[查看原尺寸](/blog/gated-deltanet/update-zh.svg)*

在这个特意选取的正交基例子里，第一列被 delta 项修正，第二列只受到 $\alpha_t$ 的全局衰减。这让两个门的职责变得可见。

## 4. “定向更新”仍然会影响相关的 key

真实表示通常不会像二维坐标轴那样整齐。取另一个查询方向 $u$，delta 校正对其读出的影响是

$$
\Delta S_tu=\beta_t e_t(k_t^\top u).
$$

当 $u$ 与 $k_t$ 正交，delta 项对这个方向没有影响；当二者相关，它也会受到影响。因此，“按 key 定向更新”需要结合表示几何来理解。一个有限状态仍然存在关联干扰，门控只是改变了如何管理这些干扰。

这个等式也给出一个有用的阅读习惯：**遇到“只改变某条记忆”的直觉描述，试着把更新式右乘另一个 key，看看条件是什么。**

还可以从在线回归理解 delta 项。设

$$
\ell(S)=\tfrac12\lVert Sk_t-v_t\rVert_2^2,
\qquad \nabla_S\ell=(Sk_t-v_t)k_t^\top.
$$

从 $\bar S_t$ 出发沿负梯度走一步，恰好得到 $\bar S_t+\beta_t(v_t-\bar S_tk_t)k_t^\top$。这里被快速更新的是**运行时状态矩阵**；生成 $q,k,v$ 和门控的网络参数仍是训练后固定的参数。这个视角解释了 fast weights 的含义，也说明“先衰减、再求残差”的顺序为何重要。[1]

## 5. 有递推依赖，为什么还能在 GPU 上高效训练

逐 token 解码时，维护一个状态很自然。训练时如果照着递推逐步执行，许多小操作难以充分使用 GPU。论文的另一个贡献是将这条规则改写成适合矩阵运算的分块形式。[1, §3.3]

把序列划分为长度为 $C$ 的块。块与块之间传递边界状态；块内用 keys 之间的关系编码更新依赖，再用矩阵运算组合结果。定义块内累计衰减

$$
\gamma_i=\prod_{r=1}^{i}\alpha_r,
\qquad
\Gamma_{ij}=\begin{cases}\gamma_i/\gamma_j&i\ge j\\0&i<j.\end{cases}
$$

令 $K\in\mathbb{R}^{C\times d_k}$ 与 $V\in\mathbb{R}^{C\times d_v}$ 的每一行分别是当前块的 key 和 value，并令 $B=\operatorname{diag}(\beta)$。原文给出的变换写入满足

$$
A\widetilde U=BV,
\qquad
A=I+\operatorname{strictLower}\bigl(B(\Gamma\odot KK^\top)\bigr).
$$

$A$ 是一个 $C\times C$ 的单位下三角矩阵。式子中的下三角结构记录了“当前位置只依赖过去”的关系。论文用 WY 表示及相应变换组织这些计算，再把初始状态的贡献并入输出与块末状态。

<img src="/blog/gated-deltanet/chunks-zh.svg" width="1200" height="1020" alt="跨块传播状态，块内利用单位下三角系统与矩阵乘法计算。">

*图 4｜分块训练的计算组织。图中的三角系统描述变换写入，完整输出还需合并进入该块的状态。[查看原尺寸](/blog/gated-deltanet/chunks-zh.svg)*

这并没有取消因果依赖，而是改变了执行组织：小步递推中的依赖，被收进块内的结构化计算里。固定 $C,d_k,d_v$ 时，总工作量可以随序列长度线性增长；实现中仍有块内计算、状态传递和中间量的成本。

由此也能看出，**更新公式是否简洁、渐近复杂度是否低、GPU 上是否快，是三个需要分别回答的问题。** 本文只解读论文的算法组织，不新增吞吐测试。

## 6. 完整的层，还需要哪些部件

状态更新式是 token mixer 的核心。原文的完整 block 还包括：[1, §3.4]

- $q,k,v$ 路径上的线性投影、短卷积与 SiLU；其中 $q,k$ 另做 L2 归一化。
- 由输入生成的衰减门与写入门，以及输出归一化、输出门和投影。
- 外层的残差结构与 SwiGLU MLP。

因此，把一条递推公式直接放进训练脚本，不能视为复现完整模型。

论文还研究了两个混合版本：H1 组合 Gated DeltaNet 和 sliding-window attention；H2 再加入 Mamba2。混合模型让递归层管理压缩状态，同时保留局部注意力的读取方式。每个注意力层仍有对应的缓存，所以“固定大小状态”的结论应明确指向递归层。

## 7. 论文的证据：纯递归有进步，混合模型还有额外收益

以下图表选取原文表 3 和表 4 的部分模型。主实验设置为 **1.3B 参数、100B FineWeb-Edu tokens、4K 训练长度**，混合模型的注意力窗口为 2K；表 4 的检索输入截断至 2K。[1, §4]

<img src="/blog/gated-deltanet/evidence-zh.svg" width="1200" height="1300" alt="作者报告：Gated DeltaNet 的常识推理均分为 55.32，检索均分为 30.6；H1 对应 56.40 和 39.0，H2 对应 56.18 和 40.1。两个指标独立绘制。">

*图 5｜原文报告值重绘，非本站复现。表 3 Avg. 是八项准确率类任务的均分，不包含 perplexity 列；表 4 Avg. 是另一组六项检索任务的均分。[查看原尺寸](/blog/gated-deltanet/evidence-zh.svg)*

相较 Mamba2，纯 Gated DeltaNet 的常识推理均分从 54.89 提升到 55.32，增加 **0.43 个百分点**；检索均分从 29.8 提升到 30.6，增加 **0.8 个百分点**。这些结果支持在该设置下组合两种更新机制有收益。

同时，纯递归版本的检索均分 30.6 仍低于 Transformer++ 的 37.0。加入注意力后的 H1、H2 分别达到 39.0 和 40.1。这组对照使“混合架构的作用”成为值得单独讨论的因素。H2 的提升不能全部归因于 gated delta rule，因为它也改变了层的组成。

这些表格没有给出上述均分的置信区间，因而不宜据此宣称细小差异具有统计显著性。它们也不是对所有模型规模、训练预算和检索任务的普遍结论。

## 8. 再读同类论文时，带上这四个问题

1. **历史存在哪里？** 是逐 token KV、有限维状态，还是两者混合？按层分开记账。
2. **状态怎样改变？** 把整体衰减、基于 key 的校正、新信息写入分别找出来。
3. **定向更新的条件是什么？** 检查 key 的归一化、相互相关性和门值范围。
4. **结果支持哪一级结论？** 区分更新规则的贡献、完整 block 的贡献和混合层组成的贡献。

读懂这篇论文，最有用的收获是能从一条更新式追踪：**旧状态被怎样保留，新关联怎样写入，这种选择又把哪些工作交给了后续的计算与架构。**

## 原始资料

1. [Gated Delta Networks: Improving Mamba2 with Delta Rule，arXiv v3](https://arxiv.org/html/2412.06464v3)；[ICLR 2025 论文](https://openreview.net/forum?id=r8H7xhYPwz)。公式对应 §2–3，主结果对应表 3、表 4。
2. [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)。线性注意力与递归形式的背景。
3. [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)。Mamba2 与 SSD 的背景。
4. [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484)。DeltaNet 与分块训练的背景。
5. [NVlabs/GatedDeltaNet 官方实现](https://github.com/NVlabs/GatedDeltaNet)。代码路径与软件支持会继续变化，本文的论文解读固定采用上述版本。

配图生成器：[`scripts/generate-gated-deltanet-figures.mjs`](https://github.com/chenhuiyu/chenhuiyu.github.io/blob/source/scripts/generate-gated-deltanet-figures.mjs)。图表数据及出处见 [sources.json](/blog/gated-deltanet/sources.json)。配图可用 `node scripts/generate-gated-deltanet-figures.mjs` 重新生成，不需要训练模型。
