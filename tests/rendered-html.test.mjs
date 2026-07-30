import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const developmentPreviewMeta =
  /<meta(?=[^>]*\bname=["']codex-preview["'])(?=[^>]*\bcontent=["']development["'])[^>]*>/i;

let worker;

test.before(async () => {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  ({ default: worker } = await import(workerUrl.href));
});

async function render(pathname) {
  return worker.fetch(
    new Request(`http://localhost${pathname}`, {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("renders development preview metadata", async () => {
  const response = await render("/");

  assert.equal(response.status, 200);
  assert.match(
    response.headers.get("content-type") ?? "",
    /^text\/html\b/i,
  );
  assert.match(await response.text(), developmentPreviewMeta);
});

test("renders every homepage topic and the latest story without a public traffic dashboard", async () => {
  const response = await render("/");
  const html = await response.text();

  assert.equal(response.status, 200);
  assert.match(html, /class="home-topics-section"/);
  assert.match(
    html,
    /href="\/series\/generative-recommendation"/,
  );
  assert.match(
    html,
    /href="\/series\/generative-recommendation-2026"/,
  );
  assert.match(
    html,
    /href="\/blog\/generative-recommendation-hands-on-zh"/,
  );
  assert.match(
    html,
    /href="\/blog\/generative-recommendation-2026-07-generalization-zh"/,
  );
  assert.match(html, /Generative recommendation/);
  assert.doesNotMatch(html, /class="visitor-pulse-section"/);
  assert.doesNotMatch(html, /busuanzi_value_site_/);
  assert.doesNotMatch(html, /vercount_value_site_/);
  assert.doesNotMatch(html, /busuanzi\.ibruce\.info/);
  assert.doesNotMatch(html, /New stories arrive here automatically/);
  assert.doesNotMatch(html, /Clear technical writing/);
  assert.match(html, /src="https:\/\/events\.vercount\.one\/js"/);
  assert.match(
    html,
    /<link rel="canonical" href="https:\/\/chenhuiyu\.github\.io\/"/,
  );
});

test("renders the interactive hands-on lab in both blog editions", async () => {
  for (const slug of [
    "generative-recommendation-hands-on-zh",
    "generative-recommendation-hands-on-en",
  ]) {
    const response = await render(`/blog/${slug}`);
    const html = await response.text();

    assert.equal(response.status, 200);
    assert.match(html, /class="generative-lab"/);
    assert.match(html, /id="semantic-id"/);
    assert.match(html, /id="constrained-decoding"/);
    assert.match(html, /id="browser-python"/);
    assert.match(html, /Run Python|运行 Python/);
    assert.match(
      html,
      /href="\/notebooks\/generative-recommendation-hands-on\.ipynb"/,
    );
    assert.match(
      html,
      /colab\.research\.google\.com\/github\/chenhuiyu\/chenhuiyu\.github\.io\/blob\/source/,
    );
  }
});

test("ships an executed, series-linked companion notebook", async () => {
  const notebook = JSON.parse(
    await readFile(
      new URL(
        "../public/notebooks/generative-recommendation-hands-on.ipynb",
        import.meta.url,
      ),
      "utf8",
    ),
  );
  const codeCells = notebook.cells.filter(
    (cell) => cell.cell_type === "code",
  );
  const errorOutputs = codeCells.flatMap((cell) =>
    cell.outputs.filter((output) => output.output_type === "error"),
  );
  const markdown = notebook.cells
    .filter((cell) => cell.cell_type === "markdown")
    .flatMap((cell) => cell.source)
    .join("");

  assert.equal(notebook.nbformat, 4);
  assert.equal(codeCells.length, 11);
  assert.ok(codeCells.every((cell) => cell.execution_count !== null));
  assert.equal(errorOutputs.length, 0);
  assert.match(markdown, /generative-recommendation-01-bpr-zh/);
  assert.match(markdown, /generative-recommendation-11-tiger-zh/);
  assert.match(markdown, /generative-recommendation-17-onerec-zh/);
});

test("renders the 2026 frontier series", async () => {
  const response = await render("/series/generative-recommendation-2026");
  const html = await response.text();

  assert.equal(response.status, 200);
  assert.match(html, /生成式推荐进入深水区/);
  assert.match(html, /DIGER/);
  assert.match(html, /GR4AD/);
});

test("keeps the Google Search Console verification file at the site root", async () => {
  const verification = await readFile(
    new URL("../public/google749d44204d33a3f0.html", import.meta.url),
    "utf8",
  );

  assert.equal(
    verification.trim(),
    "google-site-verification: google749d44204d33a3f0.html",
  );
});

test("renders article SEO and page-view metadata", async () => {
  const response = await render(
    "/blog/generative-recommendation-preface-zh",
  );
  const html = await response.text();

  assert.equal(response.status, 200);
  assert.match(html, /"@type":"BlogPosting"/);
  assert.match(html, /id="vercount_value_page_pv"/);
  assert.match(
    html,
    /<link rel="alternate" hrefLang="en" href="https:\/\/chenhuiyu\.github\.io\/blog\/generative-recommendation-preface-en"/,
  );
  assert.match(
    html,
    /<link rel="canonical" href="https:\/\/chenhuiyu\.github\.io\/blog\/generative-recommendation-preface-zh"/,
  );
});
