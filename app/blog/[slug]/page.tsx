import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { SiteFooter, SiteHeader } from "@/app/components/SiteHeader";
import { categoryLabels } from "@/lib/blog-meta";
import {
  getPost,
  getRelatedPosts,
  posts,
} from "@/lib/posts";

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

  return {
    title: `${post.title} — Huiyu Chen`,
    description: post.excerpt,
  };
}

export default async function PostPage({ params }: PageProps) {
  const { slug } = await params;
  const post = getPost(slug);
  if (!post) notFound();

  const related = getRelatedPosts(post);

  return (
    <main className="site-shell article-shell" id="top">
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
