import type { Metadata } from "next";
import { notFound } from "next/navigation";
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
    isPartOf:
      post.series === "generative-recommendation"
        ? {
            "@type": "CollectionPage",
            "@id": `${absoluteUrl(
              post.language === "zh-CN"
                ? "/series/generative-recommendation"
                : "/series/generative-recommendation/en",
            )}#series`,
            name:
              post.language === "zh-CN"
                ? "二十篇论文看懂生成式推荐的前世今生"
                : "Generative Recommendation Through 20 Papers",
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
          {post.series === "generative-recommendation" ? (
            <a
              className="article-series-link"
              href={
                post.language === "zh-CN"
                  ? "/series/generative-recommendation"
                  : "/series/generative-recommendation/en"
              }
            >
              {post.language === "zh-CN"
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

        <div className="article-body-grid">
          <aside className="article-toc" aria-label="Table of contents">
            <p>On this page</p>
            <nav>
              {post.toc.slice(0, 16).map((item) => (
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

          <div
            className="prose"
            dangerouslySetInnerHTML={{ __html: post.content }}
          />
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
