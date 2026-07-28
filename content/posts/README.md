# Writing a new bilingual post

Do not edit the generated HTML in `static-export/` or the generated
`content/authored-posts.json` file. Write Markdown here instead.

## Quick start

Generate a paired draft:

```bash
npm run post:new -- my-topic "中文标题" "English title" "NLP Insights"
```

The command uses today's Singapore date and refuses to overwrite existing
files. Set `POST_DATE=YYYY-MM-DD` before the command when you need another date.

## File pair

Every published article must have one Chinese and one English file:

```text
2026-07-24-my-post.zh-CN.md
2026-07-24-my-post.en.md
```

Both files must share the same `pairKey`. The build fails if a published
Markdown article is missing either language.

## Front matter

```yaml
---
title: "文章标题"
date: "2026-07-24"
updated: "2026-07-24"
category: "NLP Insights"
language: "zh-CN"
tags:
  - LLM
  - Recommendation
pairKey: "my-post"
slug: "my-post-zh"
excerpt: "用于博客列表和搜索结果的一句话摘要。"
draft: true
---
```

Set `draft: false` in both language files when the pair is ready. Markdown,
tables, fenced code blocks, inline math `$x$`, and display math `$$x$$` are
supported.

Run `npm run content:build` after editing. Normal development, production
builds, and static exports also compile the Markdown automatically.

## Generative recommendation series

Run `npm run content:scaffold-series` to create any missing draft files for the
20-paper series. Existing drafts are never overwritten.
