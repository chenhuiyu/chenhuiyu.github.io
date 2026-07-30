import type { Metadata } from "next";
import { RSS_ALTERNATE, SITE_SOCIAL_IMAGE } from "@/lib/site";
import { SeriesPage } from "../SeriesPage";

const description =
  "Season 2 of the generative recommendation series: seven 2026 papers on learnable Semantic IDs, cascaded reasoning, continuous tokens, item generation, page generation, advertising deployment, and generalization audits.";

export const metadata: Metadata = {
  title: "Generative Recommendation in Deep Water: The 2026 Frontier — Huiyu Chen",
  description,
  keywords: [
    "generative recommendation",
    "Semantic ID",
    "recommendation reasoning",
    "diffusion recommendation",
    "recommender systems",
    "paper review",
  ],
  alternates: {
    canonical: "/series/generative-recommendation-2026/en",
    languages: {
      "zh-CN": "/series/generative-recommendation-2026",
      en: "/series/generative-recommendation-2026/en",
      "x-default": "/series/generative-recommendation-2026/en",
    },
    types: RSS_ALTERNATE,
  },
  openGraph: {
    type: "website",
    url: "/series/generative-recommendation-2026/en",
    title: "Generative Recommendation in Deep Water: The 2026 Frontier",
    description,
    locale: "en_SG",
    alternateLocale: ["zh_CN"],
    images: [SITE_SOCIAL_IMAGE],
  },
  twitter: {
    card: "summary",
    title: "Generative Recommendation in Deep Water: The 2026 Frontier",
    description,
    images: [SITE_SOCIAL_IMAGE],
  },
};

export default function GenerativeRecommendation2026EnglishPage() {
  return <SeriesPage locale="en" />;
}
