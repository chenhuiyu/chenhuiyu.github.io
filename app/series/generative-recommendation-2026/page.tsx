import type { Metadata } from "next";
import { RSS_ALTERNATE, SITE_SOCIAL_IMAGE } from "@/lib/site";
import { SeriesPage } from "./SeriesPage";

const description =
  "生成式推荐第二季：七篇 2026 前沿论文，覆盖可学习 Semantic ID、级联推理、连续 Token、商品生成、整页推荐、广告部署与泛化审计。";

export const metadata: Metadata = {
  title: "生成式推荐进入深水区：2026 前沿第二季 — Huiyu Chen",
  description,
  keywords: [
    "生成式推荐",
    "Semantic ID",
    "推荐推理",
    "扩散推荐",
    "推荐系统",
    "论文解读",
  ],
  alternates: {
    canonical: "/series/generative-recommendation-2026",
    languages: {
      "zh-CN": "/series/generative-recommendation-2026",
      en: "/series/generative-recommendation-2026/en",
      "x-default": "/series/generative-recommendation-2026/en",
    },
    types: RSS_ALTERNATE,
  },
  openGraph: {
    type: "website",
    url: "/series/generative-recommendation-2026",
    title: "生成式推荐进入深水区：2026 前沿第二季",
    description,
    locale: "zh_CN",
    alternateLocale: ["en_SG"],
    images: [SITE_SOCIAL_IMAGE],
  },
  twitter: {
    card: "summary",
    title: "生成式推荐进入深水区：2026 前沿第二季",
    description,
    images: [SITE_SOCIAL_IMAGE],
  },
};

export default function GenerativeRecommendation2026Page() {
  return <SeriesPage locale="zh-CN" />;
}
