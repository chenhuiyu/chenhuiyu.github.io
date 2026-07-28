#!/usr/bin/env node

import { spawn } from "node:child_process";
import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const output = path.join(root, "static-export");
const port = Number(process.env.STATIC_EXPORT_PORT ?? 4179);
const origin = `http://127.0.0.1:${port}`;
const hostedSlides =
  "https://huiyu-chen-portfolio.yvettechen.chatgpt.site/talks/llm-powered-chatbot-assistants-apac-dis-2026.pdf";
const posts = JSON.parse(
  await readFile(path.join(root, "content", "posts.json"), "utf8"),
);
const authoredPosts = JSON.parse(
  await readFile(path.join(root, "content", "authored-posts.json"), "utf8"),
);

const routes = [
  "/",
  "/blog",
  "/travel",
  "/xiaohongshu",
  "/series/generative-recommendation",
  "/series/generative-recommendation/en",
  ...[...authoredPosts, ...posts].map((post) => `/blog/${post.slug}`),
];

function routeDestination(route) {
  if (route === "/") return path.join(output, "index.html");
  return path.join(output, route.slice(1), "index.html");
}

async function waitForServer() {
  let lastError;
  for (let attempt = 0; attempt < 60; attempt += 1) {
    try {
      const response = await fetch(origin);
      if (response.ok) return;
      lastError = new Error(`Server returned ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw lastError ?? new Error("Production server did not start.");
}

function cleanHtml(html) {
  return html
    .replaceAll(
      "/talks/llm-powered-chatbot-assistants-apac-dis-2026.pdf",
      hostedSlides,
    )
    .replace(
      /<meta name="codex-preview" content="development"\/?>/g,
      "",
    )
    .replace(
      /<meta name="generator" content="vinext"\/?>/g,
      '<meta name="generator" content="vinext static export"/>',
    );
}

async function renderRoute(route) {
  const response = await fetch(`${origin}${route}`);
  if (!response.ok) {
    throw new Error(`${route} returned ${response.status}`);
  }
  const destination = routeDestination(route);
  await mkdir(path.dirname(destination), { recursive: true });
  await writeFile(destination, cleanHtml(await response.text()));
}

async function renderInBatches(batchSize = 8) {
  for (let index = 0; index < routes.length; index += batchSize) {
    await Promise.all(routes.slice(index, index + batchSize).map(renderRoute));
  }
}

async function writeMetadata() {
  const siteUrl = "https://chenhuiyu.github.io";
  const sitemap = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ...routes.map(
      (route) =>
        `  <url><loc>${siteUrl}${route === "/" ? "/" : `${route}/`}</loc></url>`,
    ),
    "</urlset>",
    "",
  ].join("\n");

  await Promise.all([
    writeFile(path.join(output, ".nojekyll"), ""),
    writeFile(
      path.join(output, "robots.txt"),
      `User-agent: *\nAllow: /\nSitemap: ${siteUrl}/sitemap.xml\n`,
    ),
    writeFile(path.join(output, "sitemap.xml"), sitemap),
    cp(path.join(output, "index.html"), path.join(output, "404.html")),
  ]);
}

await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });

const server = spawn(path.join(root, "node_modules", ".bin", "vinext"), ["start"], {
  cwd: root,
  env: {
    ...process.env,
    HOSTNAME: "127.0.0.1",
    PORT: String(port),
    WRANGLER_LOG_PATH: path.join(root, ".wrangler", "wrangler.log"),
  },
  stdio: ["ignore", "pipe", "pipe"],
});

let serverLog = "";
server.stdout.on("data", (chunk) => {
  serverLog += chunk;
});
server.stderr.on("data", (chunk) => {
  serverLog += chunk;
});

try {
  await waitForServer();
  await cp(path.join(root, "dist", "client"), output, { recursive: true });
  await renderInBatches();
  await writeMetadata();
  await Promise.all([
    rm(path.join(output, ".assetsignore"), { force: true }),
    rm(path.join(output, ".vite"), { recursive: true, force: true }),
    rm(
      path.join(
        output,
        "talks",
        "llm-powered-chatbot-assistants-apac-dis-2026.pdf",
      ),
      { force: true },
    ),
  ]);
  console.log(`Exported ${routes.length} routes to ${output}`);
} catch (error) {
  if (serverLog.trim()) console.error(serverLog.trim());
  throw error;
} finally {
  server.kill("SIGTERM");
}
