#!/usr/bin/env node

import { readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import matter from "gray-matter";
import { marked } from "marked";
import markedKatex from "marked-katex-extension";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const postsDirectory = path.join(root, "content", "posts");
const outputPath = path.join(root, "content", "authored-posts.json");
const legacyPosts = JSON.parse(
  await readFile(path.join(root, "content", "posts.json"), "utf8"),
);

marked.use(
  markedKatex({
    throwOnError: false,
    output: "html",
  }),
);

function cleanText(value) {
  return String(value ?? "")
    .replace(/<[^>]+>/g, " ")
    .replaceAll("&amp;", "&")
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&quot;", '"')
    .replaceAll("&#39;", "'")
    .replace(/\s+/g, " ")
    .trim();
}

function slugify(value) {
  return String(value)
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[^\p{Letter}\p{Number}]+/gu, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 110);
}

function requireString(data, key, file) {
  const value = data[key];
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`${file}: front matter field "${key}" is required.`);
  }
  return value.trim();
}

function renderMarkdown(markdown) {
  const headingCounts = new Map();
  const toc = [];
  let html = String(marked.parse(markdown, { gfm: true }));

  html = html.replace(
    /<h([23])>([\s\S]*?)<\/h\1>/g,
    (_match, level, contents) => {
      const text = cleanText(contents);
      const base = slugify(text) || `section-${toc.length + 1}`;
      const count = headingCounts.get(base) ?? 0;
      headingCounts.set(base, count + 1);
      const id = count ? `${base}-${count + 1}` : base;
      toc.push({ id, text, level: `h${level}` });
      return `<h${level} id="${id}">${contents}</h${level}>`;
    },
  );

  html = html
    .replace(
      /<a href="(https?:\/\/[^"]+)"/g,
      '<a target="_blank" rel="noreferrer noopener" href="$1"',
    )
    .replace(
      /<img(?![^>]*\bloading=)([^>]*)>/g,
      '<img loading="lazy" decoding="async"$1>',
    );

  return { html, toc };
}

const filenames = (await readdir(postsDirectory))
  .filter((filename) => /\.(?:zh-CN|en)\.md$/.test(filename))
  .sort();

const published = [];
let draftCount = 0;

for (const filename of filenames) {
  const sourcePath = path.join(postsDirectory, filename);
  const source = await readFile(sourcePath, "utf8");
  const { data, content } = matter(source);

  if (data.draft === true) {
    draftCount += 1;
    continue;
  }

  const language = requireString(data, "language", filename);
  if (!["en", "zh-CN"].includes(language)) {
    throw new Error(`${filename}: language must be "en" or "zh-CN".`);
  }

  const title = requireString(data, "title", filename);
  const date = requireString(data, "date", filename);
  const pairKey = requireString(data, "pairKey", filename);
  const category = requireString(data, "category", filename);
  const basename = filename.replace(/\.(?:zh-CN|en)\.md$/, "");
  const slug =
    typeof data.slug === "string" && data.slug.trim()
      ? slugify(data.slug)
      : `${slugify(basename)}-${language.toLowerCase()}`;
  const { html, toc } = renderMarkdown(content);
  const plainText = cleanText(html);
  const excerpt =
    typeof data.excerpt === "string" && data.excerpt.trim()
      ? data.excerpt.trim()
      : `${plainText.slice(0, 220).trim()}${plainText.length > 220 ? "…" : ""}`;

  published.push({
    slug,
    pairKey,
    title,
    date,
    updated:
      typeof data.updated === "string" && data.updated.trim()
        ? data.updated.trim()
        : date,
    category,
    language,
    tags: Array.isArray(data.tags)
      ? data.tags.map(String)
      : typeof data.tags === "string"
        ? data.tags
            .split(",")
            .map((tag) => tag.trim())
            .filter(Boolean)
        : [],
    oldPath: "",
    content: html,
    excerpt,
    toc,
    alternateSlug: null,
    series:
      typeof data.series === "string" && data.series.trim()
        ? data.series.trim()
        : null,
    seriesOrder:
      Number.isInteger(data.seriesOrder) && data.seriesOrder >= 0
        ? data.seriesOrder
        : null,
  });
}

const legacySlugs = new Set(legacyPosts.map((post) => post.slug));
const authoredSlugs = new Set();
for (const post of published) {
  if (legacySlugs.has(post.slug) || authoredSlugs.has(post.slug)) {
    throw new Error(`Duplicate blog slug: ${post.slug}`);
  }
  authoredSlugs.add(post.slug);
}

const pairs = new Map();
for (const post of published) {
  const pair = pairs.get(post.pairKey) ?? [];
  pair.push(post);
  pairs.set(post.pairKey, pair);
}

for (const [pairKey, pair] of pairs) {
  const languages = new Set(pair.map((post) => post.language));
  if (pair.length !== 2 || !languages.has("en") || !languages.has("zh-CN")) {
    throw new Error(
      `Published pair "${pairKey}" must contain exactly one English and one Chinese post.`,
    );
  }
  pair[0].alternateSlug = pair[1].slug;
  pair[1].alternateSlug = pair[0].slug;
}

published.sort((a, b) => {
  const dateOrder = b.date.localeCompare(a.date);
  return dateOrder || b.language.localeCompare(a.language);
});

await writeFile(outputPath, `${JSON.stringify(published, null, 2)}\n`);
console.log(
  `Compiled ${published.length} published Markdown editions; ${draftCount} drafts stayed private.`,
);
