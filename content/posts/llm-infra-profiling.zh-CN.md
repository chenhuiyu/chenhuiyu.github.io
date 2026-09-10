---
title: "XProf / Nsight：从时间线找到真正瓶颈"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Infrastructure"
language: "zh-CN"
tags: ["llm-infra", "Hands-on", "Model Learning"]
pairKey: "llm-infra-profiling"
slug: "llm-infra-profiling-zh"
excerpt: "先看端到端时间线，再深入慢 kernel，用 warm-up、同步和对照实验避免性能测量陷阱。"
series: "llm-infra"
seriesOrder: 8
draft: false
---

![XProf / Nsight：从时间线找到真正瓶颈](/learning/llm-infra-profiling-zh.svg)

## 先问 GPU 为什么没在工作

Nsight Systems 适合看 CPU、CUDA、通信与跨线程时间线；Nsight Compute 深入单个 kernel 的指标；XProf 提供 accelerator 性能分析与 trace 等视图，尤其适合 XLA/JAX/TPU 工作流。三个工具关注的层次不同，不能一上来就用某个 kernel 的低利用率解释端到端延迟。

先标记一个有代表性的稳态窗口：数据加载、tokenization、CPU dispatch、GEMM、attention、collectives、同步。空白可能是等待数据、Python 开销、编译、同步或网络，不一定是设备算力不足。

## 正确计时比调参更早

GPU 操作通常异步提交，Python 调用返回时不代表 kernel 已结束。使用 CUDA events 或同步后计 wall time，warm-up 排除首次编译、初始化与缓存冷启动。CPU 则不要调用 CUDA 同步。Notebook 的 benchmark 根据设备选择正确计时，并保存分位数与环境。

```bash
# 在具备 CUDA / Nsight 的环境运行；先准备自己的 bench.py。
nsys profile --trace=cuda,nvtx,osrt -o baseline python bench.py
ncu --set basic --target-processes all python bench.py
```

Profiler 会增加开销，尤其详细 kernel 指标采集可能 replay。采样窗口与正常 benchmark 应分开，不能把被 profile 的时间直接当用户延迟。

## 从观察到可反驳的假设

如果时间线显示每个 decode step 之间有 CPU 空洞，先尝试减少 Python dispatch 或采用兼容的 graph 路径；如果 NCCL 在末尾形成长尾，检查 bucket、overlap 和慢 rank；如果 HBM 接近有效带宽而算力低，研究复用、精度与 cache 流量。每次只改一个主要因素，再复测端到端质量和延迟。

TPU/JAX 要注意首次 compilation、异步 dispatch，以及 `block_until_ready()` 的时机。XProf 中的 HLO/算子与框架层名称不总是一一对应，需要把编译后的 fusion 与源操作联系起来。

## 自测

**问题：** 开启详细 profiler 后模型慢了 5 倍，是回归吗？

<details><summary>展开答案</summary>不能直接判断。先排除采集开销、kernel replay 与同步改变。用无 profiler 的稳定 benchmark 确认真实性能，再用 trace 解释原因。</details>

## 本次真实运行结果

![编写时 CPU 实测 matmul 时间，20 次测量的 p50/p95；不是 GPU 或 TPU 基准。](/learning/llm-infra-profiling-actual.png)

编写时 CPU 实测 matmul 时间，20 次测量的 p50/p95；不是 GPU 或 TPU 基准。

## 原始资料与继续阅读

- [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [Nsight Compute profiling](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [XProf](https://openxla.org/xprof)
- [JAX asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html)

资料核对：2026-09-10。教学示例不代表生产基准；框架 API 和模型支持请以所链接版本为准。
