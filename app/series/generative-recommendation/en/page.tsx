import type { Metadata } from "next";
import { RSS_ALTERNATE, SITE_SOCIAL_IMAGE } from "@/lib/site";
import { SeriesPage } from "../SeriesPage";

const description =
  "A 20-paper story from BPR’s scoring function to OneReason’s recommendation reasoning.";

export const metadata: Metadata = {
  title: "How Recommender Systems Learned to Say the Answer — Huiyu Chen",
  description,
  keywords: [
    "generative recommendation",
    "recommender systems",
    "large language models",
    "semantic IDs",
    "paper explanations",
  ],
  alternates: {
    canonical: "/series/generative-recommendation/en",
    languages: {
      "zh-CN": "/series/generative-recommendation",
      en: "/series/generative-recommendation/en",
      "x-default": "/series/generative-recommendation/en",
    },
    types: RSS_ALTERNATE,
  },
  openGraph: {
    type: "website",
    url: "/series/generative-recommendation/en",
    title: "Generative Recommendation Through 20 Papers",
    description,
    locale: "en_SG",
    alternateLocale: ["zh_CN"],
    images: [SITE_SOCIAL_IMAGE],
  },
  twitter: {
    card: "summary",
    title: "Generative Recommendation Through 20 Papers",
    description,
    images: [SITE_SOCIAL_IMAGE],
  },
};

export default function GenerativeRecommendationSeriesEnglishPage() {
  return <SeriesPage locale="en" />;
}
