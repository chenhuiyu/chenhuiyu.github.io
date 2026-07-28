"use client";

import { useMemo, useState } from "react";
import { categoryLabels, categoryOrder } from "@/lib/blog-meta";
import type { Post } from "@/lib/posts";

type BlogCard = Pick<
  Post,
  "slug" | "title" | "date" | "category" | "language" | "excerpt" | "tags"
>;

export function BlogExplorer({ posts }: { posts: BlogCard[] }) {
  const [category, setCategory] = useState("All");
  const [language, setLanguage] = useState("All");
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return posts.filter((post) => {
      const categoryMatch = category === "All" || post.category === category;
      const languageMatch =
        language === "All" || post.language === language;
      const searchMatch =
        !normalizedQuery ||
        `${post.title} ${post.excerpt} ${post.tags.join(" ")}`
          .toLowerCase()
          .includes(normalizedQuery);
      return categoryMatch && languageMatch && searchMatch;
    });
  }, [category, language, posts, query]);

  return (
    <div className="blog-explorer">
      <div className="blog-tools">
        <div className="category-filters" aria-label="Filter by category">
          <button
            className={category === "All" ? "active" : ""}
            onClick={() => setCategory("All")}
            type="button"
          >
            All
          </button>
          {categoryOrder.map((item) => (
            <button
              className={category === item ? "active" : ""}
              key={item}
              onClick={() => setCategory(item)}
              type="button"
            >
              {categoryLabels[item]}
            </button>
          ))}
        </div>

        <div className="blog-secondary-tools">
          <label className="blog-search">
            <span className="sr-only">Search posts</span>
            <input
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search the archive…"
              type="search"
              value={query}
            />
          </label>
          <div className="language-filter" aria-label="Filter by language">
            {[
              ["All", "All"],
              ["en", "EN"],
              ["zh-CN", "中文"],
            ].map(([value, label]) => (
              <button
                className={language === value ? "active" : ""}
                key={value}
                onClick={() => setLanguage(value)}
                type="button"
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <p className="result-count">
        {filtered.length} {filtered.length === 1 ? "post" : "posts"}
      </p>

      <div className="archive-grid">
        {filtered.map((post, index) => (
          <a className="archive-card" href={`/blog/${post.slug}`} key={post.slug}>
            <div className="archive-card-top">
              <span>{String(index + 1).padStart(2, "0")}</span>
              <span>{post.language === "zh-CN" ? "中文" : "EN"}</span>
            </div>
            <p className="archive-category">
              {categoryLabels[post.category] ?? post.category}
            </p>
            <h2>{post.title}</h2>
            <p className="archive-excerpt">{post.excerpt}</p>
            <div className="archive-card-bottom">
              <time>{post.date}</time>
              <span aria-hidden="true">→</span>
            </div>
          </a>
        ))}
      </div>

      {filtered.length === 0 ? (
        <div className="empty-state">
          <span className="mini-fish" aria-hidden="true" />
          <p>No matching notes yet. Try another shelf or keyword.</p>
        </div>
      ) : null}
    </div>
  );
}
