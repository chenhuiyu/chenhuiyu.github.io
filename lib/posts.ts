import rawPosts from "@/content/posts.json";
import authoredPosts from "@/content/authored-posts.json";

export type TocItem = {
  id: string;
  text: string;
  level: "h2" | "h3";
};

export type Post = {
  slug: string;
  pairKey: string;
  title: string;
  date: string;
  updated: string;
  category: string;
  language: "en" | "zh-CN";
  tags: string[];
  oldPath: string;
  content: string;
  excerpt: string;
  toc: TocItem[];
  alternateSlug: string | null;
  series?: string | null;
  seriesOrder?: number | null;
};

export const posts = [...(authoredPosts as Post[]), ...(rawPosts as Post[])].sort(
  (a, b) => {
    const dateOrder = b.date.localeCompare(a.date);
    if (dateOrder) return dateOrder;

    const updatedOrder = b.updated.localeCompare(a.updated);
    if (updatedOrder) return updatedOrder;

    if (a.series && a.series === b.series) {
      const seriesOrder =
        (b.seriesOrder ?? Number.NEGATIVE_INFINITY) -
        (a.seriesOrder ?? Number.NEGATIVE_INFINITY);
      if (seriesOrder) return seriesOrder;
    }

    return (
      b.language.localeCompare(a.language) || b.slug.localeCompare(a.slug)
    );
  },
);

export function getPost(slug: string) {
  return posts.find((post) => post.slug === slug);
}

export function getRelatedPosts(post: Post, limit = 3) {
  return posts
    .filter(
      (candidate) =>
        candidate.slug !== post.slug &&
        (post.series
          ? candidate.series === post.series
          : candidate.category === post.category) &&
        candidate.language === post.language,
    )
    .sort((a, b) => {
      if (post.series) {
        return (a.seriesOrder ?? 999) - (b.seriesOrder ?? 999);
      }
      return b.date.localeCompare(a.date);
    })
    .slice(0, limit);
}
