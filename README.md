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

## Homepage, search performance, and SEO

The homepage reads the published post index directly. It shows the three newest
story pairs, preferring the Chinese edition for each pair, so a newly published
bilingual post appears automatically. Drafts never enter the generated index
and therefore never appear on the homepage.

Homepage topic cards are defined in `lib/topics.ts`. Add one topic definition
there when a new editorial series, season, or interactive lab should receive a
permanent homepage entrance. Counts are derived from the published post index.

The generative-recommendation hands-on article is a special bilingual post pair
that mounts `app/components/GenerativeRecommendationLab.tsx`. Its editable
Python runs in a Web Worker using the self-hosted Pyodide core under
`public/vendor/pyodide/`. After changing the pinned Pyodide version, run
`npm run vendor:pyodide`. The runtime files are generated during builds and are
not committed to the source branch.

The detailed Jupyter / Colab companion lives at
`public/notebooks/generative-recommendation-hands-on.ipynb`. It is an executed
reader-facing artifact generated from
`scripts/build-generative-recommendation-notebook.py`; rebuild and validate it
with:

```bash
npm run notebook:build
```

Google Search performance belongs in Search Console: clicks, impressions, CTR,
and average position are private reporting metrics and are intentionally not
rendered as public homepage counters. Individual article pages retain their
current-route view count through [Vercount](https://www.vercount.one/).

The permanent Google Search Console verification file is
`public/google749d44204d33a3f0.html`. Keep it at that exact root path after
verification succeeds.

Page metadata includes canonical URLs, Open Graph and Twitter cards, bilingual
`hreflang`, and Schema.org structured data. `npm run export:static` also
regenerates `robots.txt`, `sitemap.xml`, `legacy-sitemap.xml`, and `feed.xml`.
Every imported Hexo article keeps a one-to-one instant redirect from its old
URL to its new `/blog/...` URL. The canonical public origin is
`https://chenhuiyu.github.io`.

## Publish to GitHub Pages

Commit and push source changes to `source` first. When the result is ready for
production, the `Publish GitHub Pages` workflow automatically runs the same
publisher. To publish manually, run:

```bash
npm run publish:github-pages
```

The command:

1. requires a clean source working tree;
2. builds and exports every route;
3. opens `master` in an isolated Git worktree;
4. replaces only the generated Pages files;
5. asks for confirmation before pushing production.

The workflow has `contents: write` permission and runs only for `source`
updates (or a manual dispatch), so changes to generated `master` do not trigger
a publishing loop.

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
scripts/export-static.mjs static Pages, sitemap, and RSS exporter
scripts/new-post.mjs    bilingual draft generator
scripts/publish-github-pages.sh safe master publisher
scripts/vendor-pyodide.mjs refresh the browser Python runtime
scripts/build-generative-recommendation-notebook.py build and execute the full notebook
lib/topics.ts           permanent homepage topic cards
```

Generated files such as `static-export/`, `dist/`, and `node_modules/` are
ignored. Never edit generated HTML as source.

## Model learning studio (2026-09)

`/learn` and `/learn/en` connect four bilingual paths: LLM foundations (8 chapters),
LLM infrastructure (8), multimodal understanding (5), and HSTU (4). All 50 editions
are editable Markdown in `content/posts/`; `content/learning-curriculum.json` records
chapter ordering and `lib/learning/tracks.ts` defines route metadata.

Numerical React experiments live in `app/components/learning/`. Their deterministic
math is tested by `node --test tests/learning.test.mjs`. Editable Python uses the
existing self-hosted Pyodide worker. The MiniLM inspector downloads Transformers.js
3.8.1 from jsDelivr and q8 ONNX weights from Hugging Face only after a reader clicks
Run. It needs those domains reachable and runs inference locally in a worker.

Four complete notebooks are generated in both languages by:

```bash
python -m pip install nbformat matplotlib torch transformers==4.57.1 pillow
python scripts/build-learning-notebooks.py --execute
```

`--only llm-model-io`, `infra-benchmark`, `multimodal-inspection`, or
`hstu-from-scratch` selects one notebook. The generator executes code in its own
process, captures real stdout and matplotlib figures, then shares identical code
outputs between the paired editions. Published notebooks were executed on CPU;
CUDA/TPU execution is not claimed. Pretrained notebooks download model weights;
HSTU and infrastructure examples do not require external weights.

Do not regenerate without `--execute` when intending to retain executed outputs.
Generated model plots in `public/learning/*-actual.png` are captured from notebooks;
conceptual SVG diagrams are authored separately. Route changes must also update
`scripts/export-static.mjs` so GitHub Pages and the sitemap include them.

## Embedded notebooks

Every learning article mounts `EmbeddedNotebook.tsx`. Its browser edition loads a
real `.ipynb` and runs editable cells in a persistent, self-hosted Pyodide worker.
Cells share Python variables; Run all stops at the first error. Stop/restart
terminates the worker, and previous outputs are marked stale after code edits.
Core cells have a 90-second limit; the optional model-download cell has a three-minute limit. No server kernel, API key, or GPU is required.

Generate the eight bilingual browser editions with:

```bash
python scripts/build-browser-notebooks.py
node --test tests/notebook-worker.test.mjs
```

The tests execute all 16 distinct cells using actual Pyodide WebAssembly in Node
workers, including state continuity, error recovery and a fresh kernel. They are
not browser UI tests. `display_plot` renders computed data on the blog; downloaded
notebooks use a standard-Python fallback that prints the plot data.

The browser curricula deliberately use standard-library numerical experiments:
character bigram training, resource accounting, a linear reconstruction baseline,
and a fixed HSTU-style aggregator with a trained scoring head. These are not
pretrained PyTorch models. The second tab embeds the complete original PyTorch
notebooks and their saved CPU outputs; executing those full dependencies still
requires Colab/local Jupyter. The existing MiniLM page runs real pretrained ONNX
inference in the browser. The foundations notebook can also call that same model worker through an asynchronous `browser_models.embed` bridge from editable Python. This optional cell is excluded from Run all core cells and downloads weights only when explicitly run. Bridge request/response and error handling have separate protocol tests; these do not claim a browser UI test.

## Transformer observatory

`/lab/transformer` and `/lab/transformer/en` host an isolated bilingual experiment.
The permanent homepage entrance is defined in `lib/topics.ts`; existing learning
articles and notebooks are unchanged. These are experiment routes, not RSS posts.
Both routes carry canonical metadata, Schema.org LearningResource data and sitemap
language alternates.

`public/models/recall/weights.json` contains all 27,456 learned parameters of
Recall-32: 2 pre-norm causal blocks, 4 heads, width 32, learned position embeddings,
GELU MLPs, and a 15-token output vocabulary. Input examples contain 2–4 distinct
colour/animal assignments and end in the queried colour, without a final equals
sign. This is a controlled associative-recall model, not a natural-language LLM.
There is **explicit auxiliary attention supervision** for L1H1 (animal → colour)
and L2H1 (query → answer animal), plus a training-only colour probe on first-block
animal/query representations; do not describe these as spontaneously emergent
circuits. The complete training objective and fresh random validation results
are in `public/models/recall/model-card.json` and displayed on the page.

Reproduce training and independent numerical fixtures on CPU:

```bash
python -m pip install torch==2.14.0
python scripts/train-recall-transformer.py
node --test tests/observatory.test.mjs
```

The committed artifact was generated with PyTorch 2.14.0+cpu, seed 42. Exact bitwise
reproduction across PyTorch/platform versions is not guaranteed. Training is not
part of website builds; hosting needs no Python, model API, remote inference
service or GPU. Weights are fetched from this site's origin only on the experiment
page. The main homepage does not load the experiment bundle or weights.

`lib/observatory/model.ts` performs the full forward computation in JavaScript.
The three orthographically projected 3D planes display actual residual activations;
paths show the selected attention row. They are diagrams of model tensors, not
hardware execution traces. Dragging rotates the projection; a range control
provides keyboard/touch rotation, and reduced-motion settings disable path motion.
All eight independent head ablations recompute downstream layers. Exported JSON
contains the current input, intervention and tensor trace. Share links encode only
the prompt and deliberately open with the intact model.

Numerical tests compare every activation, attention weight and final probability
against independently generated PyTorch fixtures, including all eight interventions.
They also check causal invariance, masked zeros, input boundaries and recall accuracy.
These tests do not constitute browser UI testing.
