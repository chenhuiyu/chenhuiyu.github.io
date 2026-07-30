import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { GenerativeRecommendationLab } from "@/app/components/GenerativeRecommendationLab";
import { SiteFooter, SiteHeader } from "@/app/components/SiteHeader";
import { PageViewCounter } from "@/app/components/ViewCounter";
import { categoryLabels } from "@/lib/blog-meta";
import {
  getPost,
  getRelatedPosts,
  posts,
} from "@/lib/posts";
import {
  absoluteUrl,
  RSS_ALTERNATE,
  serializeJsonLd,
  SITE_SOCIAL_IMAGE,
  SITE_URL,
} from "@/lib/site";

type PageProps = {
  params: Promise<{ slug: string }>;
};

const GENERATIVE_LAB_TOC = {
  "zh-CN": [
    { id: "full-notebook", text: "完整 Notebook 与 Colab", level: "h2" as const },
    { id: "semantic-id", text: "把商品变成 Semantic ID", level: "h2" as const },
    {
      id: "residual-quantization",
      text: "Residual quantization",
      level: "h2" as const,
    },
    {
      id: "constrained-decoding",
      text: "调节生成与 Trie 约束",
      level: "h2" as const,
    },
    { id: "browser-python", text: "在浏览器运行 Python", level: "h2" as const },
  ],
  en: [
    {
      id: "full-notebook",
      text: "Full notebook and Colab",
      level: "h2" as const,
    },
    {
      id: "semantic-id",
      text: "Turn items into Semantic IDs",
      level: "h2" as const,
    },
    {
      id: "residual-quantization",
      text: "Residual quantization",
      level: "h2" as const,
    },
    {
      id: "constrained-decoding",
      text: "Tune generation and the Trie",
      level: "h2" as const,
    },
    {
      id: "browser-python",
      text: "Run Python in the browser",
      level: "h2" as const,
    },
  ],
};

export function generateStaticParams() {
  return posts.map((post) => ({ slug: post.slug }));
}

export async function generateMetadata({
  params,
}: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const post = getPost(slug);
  if (!post) return {};

  const canonical = `/blog/${post.slug}`;
  const alternatePost = post.alternateSlug
    ? getPost(post.alternateSlug)
    : undefined;
  const languageAlternates: Record<string, string> = {
    [post.language]: canonical,
  };

  if (alternatePost) {
    languageAlternates[alternatePost.language] = `/blog/${alternatePost.slug}`;
  }

  languageAlternates["x-default"] =
    post.language === "en"
      ? canonical
      : alternatePost?.language === "en"
        ? `/blog/${alternatePost.slug}`
        : canonical;

  return {
    title: `${post.title} — Huiyu Chen`,
    description: post.excerpt,
    keywords: post.tags,
    authors: [{ name: "Huiyu (Yvette) Chen", url: SITE_URL }],
    creator: "Huiyu (Yvette) Chen",
    alternates: {
      canonical,
      languages: languageAlternates,
      types: RSS_ALTERNATE,
    },
    openGraph: {
      type: "article",
      url: canonical,
      title: post.title,
      description: post.excerpt,
      publishedTime: post.date,
      modifiedTime: post.updated || post.date,
      authors: [SITE_URL],
      tags: post.tags,
      locale: post.language === "zh-CN" ? "zh_CN" : "en_SG",
      alternateLocale: [post.language === "zh-CN" ? "en_SG" : "zh_CN"],
      images: [SITE_SOCIAL_IMAGE],
    },
    twitter: {
      card: "summary",
      title: post.title,
      description: post.excerpt,
      images: [SITE_SOCIAL_IMAGE],
    },
  };
}

export default async function PostPage({ params }: PageProps) {
  const { slug } = await params;
  const post = getPost(slug);
  if (!post) notFound();

  const related = getRelatedPosts(post);
  const isGenerativeLab =
    post.pairKey === "generative-recommendation-hands-on";
  const articleToc = isGenerativeLab
    ? [...post.toc, ...GENERATIVE_LAB_TOC[post.language]]
    : post.toc;
  const seriesCollection =
    post.series === "generative-recommendation"
      ? {
          href:
            post.language === "zh-CN"
              ? "/series/generative-recommendation"
              : "/series/generative-recommendation/en",
          name:
            post.language === "zh-CN"
              ? "二十篇论文看懂生成式推荐的前世今生"
              : "Generative Recommendation Through 20 Papers",
        }
      : post.series === "generative-recommendation-2026"
        ? {
            href:
              post.language === "zh-CN"
                ? "/series/generative-recommendation-2026"
                : "/series/generative-recommendation-2026/en",
            name:
              post.language === "zh-CN"
                ? "生成式推荐进入深水区：2026 前沿第二季"
                : "Generative Recommendation in Deep Water: The 2026 Frontier",
          }
        : null;
  const canonicalUrl = absoluteUrl(`/blog/${post.slug}`);
  const articleJsonLd = {
    "@context": "https://schema.org",
    "@type": "BlogPosting",
    "@id": `${canonicalUrl}#article`,
    url: canonicalUrl,
    mainEntityOfPage: {
      "@type": "WebPage",
      "@id": canonicalUrl,
    },
    headline: post.title,
    description: post.excerpt,
    datePublished: post.date,
    dateModified: post.updated || post.date,
    inLanguage: post.language,
    articleSection: categoryLabels[post.category] ?? post.category,
    keywords: post.tags.join(", "),
    author: {
      "@id": `${SITE_URL}/#person`,
    },
    publisher: {
      "@id": `${SITE_URL}/#person`,
    },
    isPartOf: seriesCollection
      ? {
          "@type": "CollectionPage",
          "@id": `${absoluteUrl(seriesCollection.href)}#series`,
          name: seriesCollection.name,
        }
      : {
          "@id": `${SITE_URL}/#website`,
        },
  };

  return (
    <main className="site-shell article-shell" id="top">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: serializeJsonLd(articleJsonLd) }}
      />
      <SiteHeader />

      <article className="article-layout">
        <header className="article-header">
          <a className="back-link" href="/blog">
            ← All writing
          </a>
          {seriesCollection ? (
            <a
              className="article-series-link"
              href={seriesCollection.href}
            >
              {isGenerativeLab
                ? post.language === "zh-CN"
                  ? "生成式推荐 · HANDS-ON LAB"
                  : "Generative recommendation · HANDS-ON LAB"
                : post.language === "zh-CN"
                  ? `生成式推荐 · ${String(post.seriesOrder ?? 0).padStart(2, "0")}`
                  : `Generative recommendation · ${String(
                      post.seriesOrder ?? 0,
                    ).padStart(2, "0")}`}
            </a>
          ) : null}
          <div className="article-meta">
            <span>{categoryLabels[post.category] ?? post.category}</span>
            <time>{post.date}</time>
            <span>{post.language === "zh-CN" ? "中文" : "English"}</span>
            <PageViewCounter locale={post.language} />
          </div>
          <h1>{post.title}</h1>
          <p className="article-deck">{post.excerpt}</p>

          {post.alternateSlug ? (
            <a className="language-switch" href={`/blog/${post.alternateSlug}`}>
              {post.language === "zh-CN"
                ? "Read the English edition →"
                : "阅读中文版 →"}
            </a>
          ) : null}
        </header>

        <div
          className={`article-body-grid${isGenerativeLab ? " lab-article-grid" : ""}`}
        >
          <aside className="article-toc" aria-label="Table of contents">
            <p>On this page</p>
            <nav>
              {articleToc.slice(0, 16).map((item) => (
                <a
                  className={item.level === "h3" ? "toc-subitem" : ""}
                  href={`#${item.id}`}
                  key={`${item.id}-${item.text}`}
                >
                  {item.text}
                </a>
              ))}
            </nav>
          </aside>

          <div className={`prose${isGenerativeLab ? " lab-prose" : ""}`}>
            <div dangerouslySetInnerHTML={{ __html: post.content }} />
            {isGenerativeLab ? (
              <>
                <section className="lab-notebook-card" id="full-notebook">
                  <div>
                    <p>
                      {post.language === "zh-CN"
                        ? "FULL NOTEBOOK · 完整实验"
                        : "FULL NOTEBOOK · RUN IT YOURSELF"}
                    </p>
                    <h2>
                      {post.language === "zh-CN"
                        ? "从训练 BPR 到评估非法 ID，逐格运行。"
                        : "Run every step, from BPR training to invalid-ID evaluation."}
                    </h2>
                    <span>
                      {post.language === "zh-CN"
                        ? "24 个教学 cells、11 个已执行代码 cells，包含四张可视化与可失败的结果检查。无需 GPU。"
                        : "Twenty-four teaching cells, eleven executed code cells, four visualizations, and assertions that can fail. No GPU required."}
                    </span>
                  </div>
                  <div className="lab-notebook-actions">
                    <a
                      className="lab-notebook-primary"
                      href="https://colab.research.google.com/github/chenhuiyu/chenhuiyu.github.io/blob/source/public/notebooks/generative-recommendation-hands-on.ipynb"
                      rel="noreferrer"
                      target="_blank"
                    >
                      {post.language === "zh-CN"
                        ? "在 Colab 打开 ↗︎"
                        : "Open in Colab ↗︎"}
                    </a>
                    <a
                      download
                      href="/notebooks/generative-recommendation-hands-on.ipynb"
                    >
                      {post.language === "zh-CN"
                        ? "下载 .ipynb ↓"
                        : "Download .ipynb ↓"}
                    </a>
                    <a href="#semantic-id">
                      {post.language === "zh-CN"
                        ? "继续页面实验 ↓"
                        : "Continue on-page lab ↓"}
                    </a>
                  </div>
                </section>
                <GenerativeRecommendationLab locale={post.language} />
              </>
            ) : null}
          </div>
        </div>
      </article>

      {related.length ? (
        <section className="related-section">
          <p className="section-kicker">Keep reading · 继续阅读</p>
          <div className="related-grid">
            {related.map((item) => (
              <a href={`/blog/${item.slug}`} key={item.slug}>
                <span>{categoryLabels[item.category] ?? item.category}</span>
                <h2>{item.title}</h2>
                <p>{item.date}</p>
              </a>
            ))}
          </div>
        </section>
      ) : null}

      <SiteFooter />
    </main>
  );
}
