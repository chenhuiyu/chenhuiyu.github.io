# Huiyu Chen — portfolio and bilingual blog

This is the editable source for [chenhuiyu.github.io](https://chenhuiyu.github.io/).

The repository intentionally uses two branches:

- `source` — the Vinext/React source, Markdown posts, photos, and authoring tools.
- `master` — generated static files served by GitHub Pages. Do not edit it by hand.

The same source can also be continued through OpenAI Sites because
`.openai/hosting.json` is kept in the repository.

## Start developing

Requirements:

- Node.js `>=22.13.0`
- npm
- Linux or macOS

```bash
git clone --branch source https://github.com/chenhuiyu/chenhuiyu.github.io.git
cd chenhuiyu.github.io
npm ci
npm run dev
```

The local development server rebuilds published Markdown before it starts.

## Add a bilingual blog post

Create a paired Chinese and English draft:

```bash
npm run post:new -- my-topic "中文标题" "English title" "NLP Insights"
```

This creates:

```text
content/posts/YYYY-MM-DD-my-topic.zh-CN.md
content/posts/YYYY-MM-DD-my-topic.en.md
```

Write both editions, then change `draft: true` to `draft: false` in both files.
Every published post must have a real Chinese and English edition sharing the
same `pairKey`. The build rejects incomplete pairs.

Useful commands:

```bash
npm run content:build   # compile Markdown and validate bilingual pairs
npm run lint            # check source quality
npm run dev             # preview locally
npm run export:static   # generate static-export/ for GitHub Pages
```

See [content/posts/README.md](content/posts/README.md) for the full post format.

## Publish to GitHub Pages

Commit and push source changes to `source` first. When the result is ready for
production, run:

```bash
npm run publish:github-pages
```

The command:

1. requires a clean source working tree;
2. builds and exports every route;
3. opens `master` in an isolated Git worktree;
4. replaces only the generated Pages files;
5. asks for confirmation before pushing production.

For an already-approved non-interactive agent run:

```bash
PUBLISH_CONFIRM=YES npm run publish:github-pages
```

The old Hexo source is preserved on
`archive/hexo-source-before-portfolio-2026-07-28`, and the old deployed Hexo
site remains on `archive/hexo-before-portfolio-2026-07-24`.

## Project map

```text
app/                    pages and components
content/posts/          editable bilingual Markdown posts
content/posts.json      migrated legacy bilingual archive
content/travel.json     travel timeline and locations
content/xiaohongshu.json curated Xiaohongshu entries
public/                 photos, travel images, and talk materials
scripts/build-posts.mjs Markdown compiler and bilingual validation
scripts/export-static.mjs static GitHub Pages exporter
scripts/new-post.mjs    bilingual draft generator
scripts/publish-github-pages.sh safe master publisher
```

Generated files such as `static-export/`, `dist/`, and `node_modules/` are
ignored. Never edit generated HTML as source.
