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

## 从零开始的完整讲解

### GPU 为什么适合矩阵，又为什么也会闲着

CPU 擅长复杂控制与低延迟串行工作；GPU 将大量并行执行资源用于规则的大规模计算。GPU 上有执行线程、寄存器、片上共享存储与较大的设备内存，访问不同层级的代价不同。数据是否能被重复使用，往往与算术总量同样重要。

TPU 也为张量计算设计专门硬件，但其执行、编译和存储组织不能简单套用 CUDA 的线程模型。学习时抓住共同问题：矩阵单元怎样持续获得数据，片上容量怎样限制分块，跨设备如何通信。具体代际规格应查看厂商资料。

### 用 roofline 判断先优化搬运还是算术

Arithmetic intensity 是每读取或写入一个 byte 所完成的运算数量，常以 FLOPs/byte 表示。Roofline 的简单上界为 `min(峰值算力, 内存带宽*算术强度)`。例如某假想设备峰值 100 TFLOP/s、有效带宽 1 TB/s，强度 10 FLOPs/byte 的 kernel 即使理想执行，带宽上界也只有 10 TFLOP/s。

这不是设备实测值，而是单位分析示例。若 FLOP 统计与 peak 使用不同精度或稀疏假设，图会失去意义。通信、launch 和依赖导致的空闲也可能让实测远低于这条简单上界。

### Tiling 为什么能减少反复读取

矩阵乘法 `C=A@B` 中，一个 A 元素会与多个 B 元素相乘。若每次都从外部内存重新读取，会浪费带宽。Tiling 把一小块数据放入更近的存储，再复用多次，增加算术强度。

但 tile 不是越大越好：它可能耗尽寄存器或共享内存，减少同时驻留的工作，甚至产生 spill。性能优化常需要在复用、并发和资源占用之间找平衡。只背“提高 occupancy”也不够，高 occupancy 不保证数据访问高效。

### Fusion 怎样减少中间张量

假设顺序计算加 bias、激活、再缩放，分三个 kernel 可能把中间结果反复写入和读出设备内存。融合后可以在一次执行中保留局部值，降低中间流量与 launch 次数。这对很多逐元素操作有价值。

然而融合可能增大寄存器压力，或使原本高效的库算子失去优化路径。应比较完整子图耗时与正确性，不能只因为 kernel 数量少就判定更快。

### 低精度计算必须同时考虑累加

输入采用较低精度，不一定意味着累加也用相同精度。长点积的累加误差可能影响结果，因此实现经常对输入、累加器和输出采用不同格式。BF16 与 FP16 的指数范围和精度分配不同，不能说它们都占 2 bytes 就完全等价。

对优化后的 kernel，先比较小 shape、边界 shape、非整除尺寸与极端数值。检查容差与数据类型是否符合任务，然后再进行预热和设备同步后的计时。错误但很快的内核没有实际价值。

### 学 CUDA/Triton 与 TPU 优化的共同路线

先用普通 Python 或张量库写正确参考，实现向量加法、归约和矩阵乘法，再观察内存访问、分块及融合。对于 TPU，理解编译图、shape、布局和分片如何改变生成执行；对于 GPU，结合 kernel 级指标与系统时间线。两条路线都需要以测量为依据，而不是仅追逐理论 FLOPs。

下方 Notebook 的 CPU 循环用于理解计算量和计时噪声，不代表 GPU 内核。将矩阵边长翻倍，普通三重循环的乘加数量约增为 8 倍；实测时间受解释器与设备状态影响，观察差异本身也是实验的一部分。

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
