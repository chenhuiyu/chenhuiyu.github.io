#!/usr/bin/env node

import { access, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const postsDir = path.join(root, "content", "posts");
const [slugInput, zhTitle, enTitle, category = "NLP Insights"] =
  process.argv.slice(2);

function fail(message) {
  console.error(message);
  process.exit(1);
}

function singaporeDate() {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Singapore",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(new Date());
  const value = Object.fromEntries(parts.map(({ type, value }) => [type, value]));
  return `${value.year}-${value.month}-${value.day}`;
}

function yamlString(value) {
  return JSON.stringify(value);
}

function postTemplate({ title, language, slug, pairKey, date }) {
  const opening =
    language === "zh-CN"
      ? "在这里写一句话说明这篇文章要解决的问题。"
      : "State the question this article will answer in one sentence.";

  return `---
title: ${yamlString(title)}
date: ${yamlString(date)}
updated: ${yamlString(date)}
category: ${yamlString(category)}
language: ${yamlString(language)}
tags: []
pairKey: ${yamlString(pairKey)}
slug: ${yamlString(slug)}
excerpt: ${yamlString(opening)}
draft: true
---

## 30 秒看懂本文

${opening}

## 问题从哪里来 / Where the problem comes from

## 方法与直觉 / Method and intuition

## 训练与推理 / Training and inference

## 实验与局限 / Evidence and limitations

## 下一步 / What comes next
`;
}

if (!slugInput || !zhTitle || !enTitle) {
  fail(
    'Usage: npm run post:new -- <slug> "<Chinese title>" "<English title>" [category]',
  );
}

const pairKey = slugInput
  .trim()
  .toLowerCase()
  .replace(/[^a-z0-9]+/g, "-")
  .replace(/^-+|-+$/g, "");

if (!pairKey) {
  fail("The slug must contain at least one ASCII letter or number.");
}

const date = process.env.POST_DATE || singaporeDate();
if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
  fail("POST_DATE must use YYYY-MM-DD.");
}

const files = [
  {
    file: path.join(postsDir, `${date}-${pairKey}.zh-CN.md`),
    title: zhTitle,
    language: "zh-CN",
    slug: `${pairKey}-zh`,
  },
  {
    file: path.join(postsDir, `${date}-${pairKey}.en.md`),
    title: enTitle,
    language: "en",
    slug: `${pairKey}-en`,
  },
];

await mkdir(postsDir, { recursive: true });

for (const { file } of files) {
  try {
    await access(file);
    fail(`Refusing to overwrite existing file: ${path.relative(root, file)}`);
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }
}

await Promise.all(
  files.map(({ file, ...post }) =>
    writeFile(file, postTemplate({ ...post, pairKey, date }), "utf8"),
  ),
);

console.log("Created bilingual drafts:");
for (const { file } of files) {
  console.log(`- ${path.relative(root, file)}`);
}
