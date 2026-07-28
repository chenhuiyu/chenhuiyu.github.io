import type { Metadata } from "next";
import { SeriesPage } from "../SeriesPage";

export const metadata: Metadata = {
  title: "How Recommender Systems Learned to Say the Answer — Huiyu Chen",
  description:
    "A 20-paper story from BPR’s scoring function to OneReason’s recommendation reasoning.",
};

export default function GenerativeRecommendationSeriesEnglishPage() {
  return <SeriesPage locale="en" />;
}
