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
const siteUrl = "https://chenhuiyu.github.io";
const hostedSlides =
  "https://huiyu-chen-portfolio.yvettechen.chatgpt.site/talks/llm-powered-chatbot-assistants-apac-dis-2026.pdf";
const posts = JSON.parse(
  await readFile(path.join(root, "content", "posts.json"), "utf8"),
);
const authoredPosts = JSON.parse(
  await readFile(path.join(root, "content", "authored-posts.json"), "utf8"),
);
const allPosts = [...authoredPosts, ...posts];
const postsBySlug = new Map(allPosts.map((post) => [post.slug, post]));
const legacyPosts = allPosts.filter(
  (post) => typeof post.oldPath === "string" && post.oldPath.trim(),
);

const routes = [
  "/",
  "/blog",
  "/travel",
  "/xiaohongshu",
  "/series/generative-recommendation",
  "/series/generative-recommendation/en",
  "/series/generative-recommendation-2026",
  "/series/generative-recommendation-2026/en",
  ...allPosts.map((post) => `/blog/${post.slug}`),
];

function routeDestination(route) {
  if (route === "/") return path.join(output, "index.html");
  return path.join(output, route.slice(1), "index.html");
}

function legacyRouteDestination(route) {
  const pathname = decodeURIComponent(
    new URL(route, "https://legacy.example").pathname,
  );
  const relativePath = pathname.replace(/^\/+/, "").replace(/\/+$/, "");
  const destination = path.resolve(output, relativePath, "index.html");
  const outputPrefix = `${path.resolve(output)}${path.sep}`;

  if (!relativePath || !destination.startsWith(outputPrefix)) {
    throw new Error(`Unsafe legacy path: ${route}`);
  }
  return destination;
}

function validateLegacyMappings() {
  const seen = new Set();

  for (const post of legacyPosts) {
    if (!post.oldPath.startsWith("/")) {
      throw new Error(`Legacy path must start with "/": ${post.oldPath}`);
    }
    if (seen.has(post.oldPath)) {
      throw new Error(`Duplicate legacy path: ${post.oldPath}`);
    }
    seen.add(post.oldPath);
    legacyRouteDestination(post.oldPath);
  }
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

function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}

function routeUrl(siteUrl, route) {
  return new URL(route === "/" ? "/" : route, `${siteUrl}/`).toString();
}

function postLanguageAlternates(siteUrl, post) {
  const alternate = post.alternateSlug
    ? postsBySlug.get(post.alternateSlug)
    : undefined;
  const links = [
    {
      language: post.language,
      href: routeUrl(siteUrl, `/blog/${post.slug}`),
    },
  ];

  if (alternate) {
    links.push({
      language: alternate.language,
      href: routeUrl(siteUrl, `/blog/${alternate.slug}`),
    });
  }

  const english =
    post.language === "en"
      ? post
      : alternate?.language === "en"
        ? alternate
        : post;
  links.push({
    language: "x-default",
    href: routeUrl(siteUrl, `/blog/${english.slug}`),
  });
  return links;
}

function sitemapAlternates(siteUrl, route) {
  const seriesRoot = route.endsWith("/en") ? route.slice(0, -3) : route;
  if (
    seriesRoot === "/series/generative-recommendation" ||
    seriesRoot === "/series/generative-recommendation-2026"
  ) {
    return [
      {
        language: "en",
        href: routeUrl(siteUrl, `${seriesRoot}/en`),
      },
      {
        language: "zh-CN",
        href: routeUrl(siteUrl, seriesRoot),
      },
      {
        language: "x-default",
        href: routeUrl(siteUrl, `${seriesRoot}/en`),
      },
    ];
  }

  if (route.startsWith("/blog/")) {
    const post = postsBySlug.get(route.slice("/blog/".length));
    return post ? postLanguageAlternates(siteUrl, post) : [];
  }

  return [];
}

function sitemapRouteMeta(route, latestModified, latestSeriesModified) {
  if (route === "/") {
    return { lastmod: latestModified, changefreq: "weekly", priority: "1.0" };
  }
  if (route === "/blog") {
    return { lastmod: latestModified, changefreq: "daily", priority: "0.9" };
  }
  if (route.startsWith("/series/generative-recommendation")) {
    return {
      lastmod: latestSeriesModified,
      changefreq: "weekly",
      priority: "0.9",
    };
  }
  if (route.startsWith("/blog/")) {
    const post = postsBySlug.get(route.slice("/blog/".length));
    return {
      lastmod: post?.updated || post?.date,
      changefreq: "monthly",
      priority: "0.7",
    };
  }
  return { changefreq: "monthly", priority: "0.6" };
}

function formatRssDate(date) {
  return new Date(`${date}T00:00:00Z`).toUTCString();
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

async function writeLegacyRedirect(post) {
  const destination = legacyRouteDestination(post.oldPath);
  const targetUrl = routeUrl(siteUrl, `/blog/${post.slug}`);
  const language = post.language === "zh-CN" ? "zh-CN" : "en";
  const title =
    language === "zh-CN"
      ? `${post.title}｜文章已迁移`
      : `${post.title} | Article moved`;
  const message =
    language === "zh-CN"
      ? "文章已迁移到新地址，正在跳转。"
      : "This article has moved. Redirecting to its new address.";
  const linkLabel = language === "zh-CN" ? "打开新地址" : "Open the new address";
  const html = [
    "<!doctype html>",
    `<html lang="${language}">`,
    "<head>",
    '  <meta charset="utf-8">',
    `  <meta http-equiv="refresh" content="0; url=${escapeXml(targetUrl)}">`,
    `  <link rel="canonical" href="${escapeXml(targetUrl)}">`,
    `  <title>${escapeXml(title)}</title>`,
    "</head>",
    "<body>",
    "  <main>",
    `    <h1>${escapeXml(post.title)}</h1>`,
    `    <p>${escapeXml(message)}</p>`,
    `    <p><a href="${escapeXml(targetUrl)}">${escapeXml(linkLabel)}</a></p>`,
    "  </main>",
    "</body>",
    "</html>",
    "",
  ].join("\n");

  await mkdir(path.dirname(destination), { recursive: true });
  await writeFile(destination, html);
}

async function writeLegacyRedirects(batchSize = 16) {
  for (let index = 0; index < legacyPosts.length; index += batchSize) {
    await Promise.all(
      legacyPosts.slice(index, index + batchSize).map(writeLegacyRedirect),
    );
  }
}

async function writeMetadata() {
  const latestModified = allPosts
    .map((post) => post.updated || post.date)
    .sort()
    .at(-1);
  const latestSeriesModified = allPosts
    .filter((post) => post.series?.startsWith("generative-recommendation"))
    .map((post) => post.updated || post.date)
    .sort()
    .at(-1);
  const sitemap = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml">',
    ...routes.map((route) => {
      const meta = sitemapRouteMeta(
        route,
        latestModified,
        latestSeriesModified,
      );
      const alternateLinks = sitemapAlternates(siteUrl, route)
        .map(
          ({ language, href }) =>
            `    <xhtml:link rel="alternate" hreflang="${escapeXml(
              language,
            )}" href="${escapeXml(href)}" />`,
        )
        .join("\n");
      return [
        "  <url>",
        `    <loc>${escapeXml(routeUrl(siteUrl, route))}</loc>`,
        meta.lastmod ? `    <lastmod>${escapeXml(meta.lastmod)}</lastmod>` : "",
        `    <changefreq>${meta.changefreq}</changefreq>`,
        `    <priority>${meta.priority}</priority>`,
        alternateLinks,
        "  </url>",
      ]
        .filter(Boolean)
        .join("\n");
    }),
    "</urlset>",
    "",
  ].join("\n");
  const legacySitemap = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ...legacyPosts.map((post) =>
      [
        "  <url>",
        `    <loc>${escapeXml(routeUrl(siteUrl, post.oldPath))}</loc>`,
        "  </url>",
      ].join("\n"),
    ),
    "</urlset>",
    "",
  ].join("\n");
  const feedPosts = [...allPosts]
    .sort(
      (a, b) =>
        b.date.localeCompare(a.date) ||
        b.language.localeCompare(a.language),
    )
    .slice(0, 30);
  const feed = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom" xmlns:dc="http://purl.org/dc/elements/1.1/">',
    "  <channel>",
    "    <title>Huiyu Chen — Writing</title>",
    `    <link>${siteUrl}/blog</link>`,
    "    <description>Notes on models, systems, and being human — in English and Chinese.</description>",
    `    <atom:link href="${siteUrl}/feed.xml" rel="self" type="application/rss+xml" />`,
    "    <language>en-SG</language>",
    `    <lastBuildDate>${formatRssDate(latestModified)}</lastBuildDate>`,
    ...feedPosts.flatMap((post) => {
      const url = routeUrl(siteUrl, `/blog/${post.slug}`);
      return [
        "    <item>",
        `      <title>${escapeXml(post.title)}</title>`,
        `      <link>${escapeXml(url)}</link>`,
        `      <guid isPermaLink="true">${escapeXml(url)}</guid>`,
        `      <pubDate>${formatRssDate(post.date)}</pubDate>`,
        `      <dc:language>${escapeXml(post.language)}</dc:language>`,
        `      <category>${escapeXml(post.category)}</category>`,
        `      <description>${escapeXml(post.excerpt)}</description>`,
        "    </item>",
      ];
    }),
    "  </channel>",
    "</rss>",
    "",
  ].join("\n");

  await Promise.all([
    writeFile(path.join(output, ".nojekyll"), ""),
    writeFile(
      path.join(output, "robots.txt"),
      [
        "User-agent: *",
        "Allow: /",
        `Sitemap: ${siteUrl}/sitemap.xml`,
        `Sitemap: ${siteUrl}/legacy-sitemap.xml`,
        "",
      ].join("\n"),
    ),
    writeFile(path.join(output, "sitemap.xml"), sitemap),
    writeFile(path.join(output, "legacy-sitemap.xml"), legacySitemap),
    writeFile(path.join(output, "feed.xml"), feed),
    cp(path.join(output, "index.html"), path.join(output, "404.html")),
  ]);
}

await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
validateLegacyMappings();

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
  await writeLegacyRedirects();
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
  console.log(
    `Exported ${routes.length} routes and ${legacyPosts.length} legacy redirects to ${output}`,
  );
} catch (error) {
  if (serverLog.trim()) console.error(serverLog.trim());
  throw error;
} finally {
  server.kill("SIGTERM");
}
