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

test("renders the homepage series and latest story without a public traffic dashboard", async () => {
  const response = await render("/");
  const html = await response.text();

  assert.equal(response.status, 200);
  assert.match(html, /class="home-series-section"/);
  assert.match(
    html,
    /href="\/blog\/generative-recommendation-preface-zh"/,
  );
  assert.match(
    html,
    /href="\/blog\/generative-recommendation-20-onereason-zh"/,
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
