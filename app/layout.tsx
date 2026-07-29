import type { Metadata } from "next";
import { Geist, Geist_Mono, Newsreader } from "next/font/google";
import {
  LINKEDIN_URL,
  RSS_ALTERNATE,
  serializeJsonLd,
  SITE_DESCRIPTION,
  SITE_NAME,
  SITE_SOCIAL_IMAGE,
  SITE_TITLE,
  SITE_URL,
} from "@/lib/site";
import "katex/dist/katex.min.css";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const newsreader = Newsreader({
  variable: "--font-serif",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: SITE_TITLE,
  description: SITE_DESCRIPTION,
  applicationName: SITE_NAME,
  authors: [{ name: "Huiyu (Yvette) Chen", url: SITE_URL }],
  creator: "Huiyu (Yvette) Chen",
  publisher: SITE_NAME,
  keywords: [
    "Huiyu Chen",
    "Yvette Chen",
    "machine learning engineer",
    "multimodal LLM",
    "large language models",
    "generative recommendation",
    "recommender systems",
    "NLP",
  ],
  alternates: {
    canonical: "/",
    types: RSS_ALTERNATE,
  },
  openGraph: {
    type: "website",
    url: "/",
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
    siteName: SITE_NAME,
    locale: "en_SG",
    alternateLocale: ["zh_CN"],
    images: [
      {
        url: SITE_SOCIAL_IMAGE,
        width: 1085,
        height: 1450,
        alt: "Portrait of Huiyu Chen",
      },
    ],
  },
  twitter: {
    card: "summary",
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
    images: [SITE_SOCIAL_IMAGE],
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-image-preview": "large",
      "max-snippet": -1,
      "max-video-preview": -1,
    },
  },
  other: {
    "codex-preview": "development",
  },
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const identityJsonLd = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Person",
        "@id": `${SITE_URL}/#person`,
        name: "Huiyu Chen",
        alternateName: "Yvette Chen",
        url: SITE_URL,
        image: `${SITE_URL}${SITE_SOCIAL_IMAGE}`,
        jobTitle: "Machine Learning Engineer",
        worksFor: {
          "@type": "Organization",
          name: "Meta",
        },
        sameAs: [LINKEDIN_URL],
      },
      {
        "@type": "WebSite",
        "@id": `${SITE_URL}/#website`,
        url: `${SITE_URL}/`,
        name: SITE_NAME,
        description: SITE_DESCRIPTION,
        inLanguage: ["en", "zh-CN"],
        author: {
          "@id": `${SITE_URL}/#person`,
        },
      },
    ],
  };

  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} ${newsreader.variable} antialiased`}
      >
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: serializeJsonLd(identityJsonLd) }}
        />
        {children}
        <script
          async
          src="https://busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js"
        />
        <script defer src="https://events.vercount.one/js" />
      </body>
    </html>
  );
}
