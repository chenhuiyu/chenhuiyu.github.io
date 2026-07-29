# Agent handoff

This repository has two distinct responsibilities:

- `source` is the only development branch.
- `master` contains generated static files for `https://chenhuiyu.github.io/`.

Never implement features or blog edits directly on `master`.

## Before editing

1. Work from `source`.
2. Run `npm ci` when dependencies are missing.
3. Read `README.md` and `content/posts/README.md`.
4. Preserve `.openai/hosting.json`; it connects this source to the existing
   OpenAI Sites project.

## Blog rules

- New posts are Markdown files in `content/posts/`.
- Every published article needs one `.zh-CN.md` file and one `.en.md` file.
- Both files use the same `pairKey` and different slugs.
- Keep both files at `draft: true` until both editions are complete.
- Do not hand-edit `content/authored-posts.json`; it is generated.
- Do not hand-edit files in `static-export/` or `master`.
- Do not maintain a hard-coded homepage article list. The homepage derives the
  newest published story pairs from `lib/posts.ts`.
- Keep canonical URLs on `https://chenhuiyu.github.io`; update page metadata,
  structured data, RSS, and sitemap behavior together when routes change.
- Preserve the Vercount element IDs. They are filled by the shared deferred
  script in `app/layout.tsx`.

Use:

```bash
npm run post:new -- slug "中文标题" "English title" "NLP Insights"
npm run content:build
npm run lint
```

## Validation and publishing

- For Sites work, follow the Sites lifecycle and preview instructions.
- For GitHub Pages, run `npm run export:static` and inspect the result.
- Pushing `source` triggers `.github/workflows/publish-github-pages.yml`, which
  builds the source and safely updates generated `master`.
- `npm run publish:github-pages` replaces production `master`; run it only
  after the user explicitly approves publishing.
- Preserve unrelated user changes and all existing photos/content.

When handing off, report the edited Markdown pair, validation performed, source
commit, and whether production was published.
