---
title: "GPU、TPU 与 Kernel：为什么 FLOPs 解释不了速度？"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Infrastructure"
language: "zh-CN"
tags: ["llm-infra", "Hands-on", "Model Learning"]
pairKey: "llm-infra-hardware"
slug: "llm-infra-hardware-zh"
excerpt: "用 Roofline 理解 HBM、片上存储、矩阵单元、tiling 与 fusion，再看 Triton/Pallas 的作用。"
series: "llm-infra"
seriesOrder: 7
draft: false
---

![GPU、TPU 与 Kernel：为什么 FLOPs 解释不了速度？](/learning/llm-infra-hardware-zh.svg)

## 一次矩阵乘穿过哪些存储？

GPU 的计算包含线程/warp 调度、SM 上执行资源、寄存器、共享内存与 HBM；Tensor Cores 加速适合的矩阵操作，不替代所有通用运算。TPU 包含矩阵单元、向量/标量单元与片上内存，通过编译器把程序映射到阵列。不同代际硬件结构与精度支持不同，不能把“GPU 灵活、TPU 只能矩阵乘”当完整解释。

昂贵的往往是搬数据。若每次操作都把中间结果写回 HBM，下一个 kernel 又读回来，理论 FLOPs 再低也可能很慢。Tiling 将一块输入载入片上存储后反复使用；fusion 减少中间写回，但可能占用更多寄存器或限制并行度。

## Roofline 的两个上界

算术强度 $I=\mathrm{FLOPs}/\mathrm{bytes}$。简化吞吐上界是 $\min(P_{peak}, BW\times I)$。对应时间下界为 $\max(F/P_{peak},M/BW)$。这是假设能有效重叠与理想利用的边界，不是预测器；launch、同步、负载不均和通信会让实际更慢。

```python
flops = 2 * 1024**3
bytes_moved = 3 * 1024**2 * 2  # 教学假设：每矩阵读/写一次
intensity = flops / bytes_moved
print('FLOP/byte:', intensity)
```

真实 GEMM 有缓存复用、不同布局和 dtype，实际搬运不能总按这个简单三矩阵公式估计。

## 写 kernel 之前先确认形状

Triton 教程的 tiled matmul 展示 program instances、block pointers 与 masked loads 等思想。TPU 上可用 JAX/Pallas 进行更接近硬件的 kernel 编写，具体后端限制需查文档。优化顺序是：数值正确，尾块安全，stride/layout 正确，再扫描 block sizes、warps 和 pipeline stages。

先对随机 shape 和非整除维度比较高精度 reference，再 benchmark。一个 1024×1024 上快的 kernel，可能在 decode 的细长矩阵上更慢。报告 TFLOP/s 时写清乘加算 2 FLOPs、是否含数据搬运，以及实际精度。

## 自测

**问题：** 一个 kernel 只有峰值算力的 10%，就一定写得差吗？

<details><summary>展开答案</summary>不一定。它可能处在低算术强度的带宽上界附近。先比较适用的 Roofline，再看访存效率、占用率与同步；高 occupancy 本身也不是最终目标。</details>

## 原始资料与继续阅读

- [TPU architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)
- [Triton matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Pallas](https://docs.jax.dev/en/latest/pallas/index.html)

资料核对：2026-09-10。教学示例不代表生产基准；框架 API 和模型支持请以所链接版本为准。
