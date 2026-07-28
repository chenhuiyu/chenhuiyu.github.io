#!/usr/bin/env node

import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const postsDirectory = path.join(root, "content", "posts");
const series = JSON.parse(
  await readFile(
    path.join(root, "content", "generative-recommendation-series.json"),
    "utf8",
  ),
);

function slugify(value) {
  return String(value)
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[^\p{Letter}\p{Number}]+/gu, "-")
    .replace(/^-+|-+$/g, "");
}

function yamlString(value) {
  return JSON.stringify(String(value));
}

function chineseBody(episode) {
  return `> 核心论文：**${episode.paper}**
>
> 本篇任务：${episode.role["zh-CN"]}

## 30 秒看懂本文

<!-- 用三句话写清：上一代的问题、本文的方法、最重要的证据。 -->

## 上一篇留下了什么问题

<!-- 从系列主线自然接入，不要从论文摘要重新开始。 -->

## 用一个八件商品的玩具例子理解方法

<!-- 固定用户行为：网球 → 攀岩 → 羽毛球 → 游泳。 -->

## 模型从输入到输出发生了什么

<!-- 给出输入、tensor shape、关键模块、输出和一张数据流图。 -->

## 核心公式逐项拆解

<!-- 只选一个核心公式，把每个符号映射到玩具例子。 -->

## 训练和推理分别怎么做

<!-- 写清训练样本构造、目标函数、负采样或解码过程、复杂度。 -->

## 实验究竟证明了什么

<!-- 重绘一张主实验或 ablation；区分证据与作者解释。 -->

## 它失败在哪里

<!-- 写出假设、边界、工程代价，以及论文没有证明什么。 -->

## 下一篇为什么会出现

<!-- 用尚未解决的问题连接下一篇。 -->
`;
}

function englishBody(episode) {
  return `> Core paper: **${episode.paper}**
>
> Narrative job: ${episode.role.en}

## Understand the paper in 30 seconds

<!-- Three sentences: the previous limitation, this method, and the strongest evidence. -->

## What problem did the previous paper leave behind?

<!-- Continue the series narrative instead of restarting from the abstract. -->

## Understand the method with eight toy items

<!-- Reuse the fixed history: tennis → climbing → badminton → swimming. -->

## What happens from model input to output?

<!-- Specify inputs, tensor shapes, modules, outputs, and one data-flow figure. -->

## Unpack the central equation

<!-- Keep one equation and connect every symbol to the toy example. -->

## Training versus inference

<!-- Cover sample construction, objectives, negatives or decoding, and complexity. -->

## What do the experiments actually prove?

<!-- Redraw one decisive result or ablation and separate evidence from interpretation. -->

## Where does it fail?

<!-- State assumptions, boundaries, engineering cost, and what remains unproven. -->

## Why does the next paper need to exist?

<!-- End with the unresolved problem that motivates the next episode. -->
`;
}

await mkdir(postsDirectory, { recursive: true });
let created = 0;
let skipped = 0;

for (const episode of series.episodes) {
  const order = String(episode.order).padStart(2, "0");
  const paperSlug = slugify(episode.paper);
  const pairKey = `generative-recommendation-${order}-${paperSlug}`;

  for (const language of ["zh-CN", "en"]) {
    const filename = `${order}-${paperSlug}.${language}.md`;
    const destination = path.join(postsDirectory, filename);
    if (existsSync(destination)) {
      skipped += 1;
      continue;
    }

    const title =
      language === "zh-CN"
        ? `${episode.title["zh-CN"]}——${episode.paper}`
        : `${episode.title.en} — ${episode.paper}`;
    const excerpt = episode.role[language];
    const slug = `${pairKey}-${language === "zh-CN" ? "zh" : "en"}`;
    const body =
      language === "zh-CN" ? chineseBody(episode) : englishBody(episode);
    const source = `---
title: ${yamlString(title)}
date: "2026-07-24"
updated: "2026-07-24"
category: "Generative Recommendation"
language: ${yamlString(language)}
tags:
  - Generative Recommendation
  - Recommender Systems
  - ${yamlString(episode.paper)}
pairKey: ${yamlString(pairKey)}
slug: ${yamlString(slug)}
excerpt: ${yamlString(excerpt)}
series: "generative-recommendation"
seriesOrder: ${episode.order}
draft: true
---

${body}`;
    await writeFile(destination, source);
    created += 1;
  }
}

console.log(
  `Created ${created} series drafts; preserved ${skipped} existing files.`,
);
