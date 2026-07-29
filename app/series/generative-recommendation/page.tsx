import type { Metadata } from "next";
import { RSS_ALTERNATE, SITE_SOCIAL_IMAGE } from "@/lib/site";
import { SeriesPage } from "./SeriesPage";

const description =
  "从 BPR 的打分器，到 OneReason 的推荐推理：20 篇论文串起生成式推荐的完整故事。";

export const metadata: Metadata = {
  title: "推荐系统如何学会说出答案 — Huiyu Chen",
  description,
  keywords: [
    "生成式推荐",
    "推荐系统",
    "大语言模型",
    "Semantic ID",
    "论文解读",
  ],
  alternates: {
    canonical: "/series/generative-recommendation",
    languages: {
      "zh-CN": "/series/generative-recommendation",
      en: "/series/generative-recommendation/en",
      "x-default": "/series/generative-recommendation/en",
    },
    types: RSS_ALTERNATE,
  },
  openGraph: {
    type: "website",
    url: "/series/generative-recommendation",
    title: "二十篇论文看懂生成式推荐的前世今生",
    description,
    locale: "zh_CN",
    alternateLocale: ["en_SG"],
    images: [SITE_SOCIAL_IMAGE],
  },
  twitter: {
    card: "summary",
    title: "二十篇论文看懂生成式推荐的前世今生",
    description,
    images: [SITE_SOCIAL_IMAGE],
  },
};

export default function GenerativeRecommendationSeriesPage() {
  return <SeriesPage locale="zh-CN" />;
}
