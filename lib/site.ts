export const SITE_URL = "https://chenhuiyu.github.io";
export const SITE_NAME = "Huiyu Chen";
export const SITE_TITLE = "Huiyu Chen — Machine Learning Engineer at Meta";
export const SITE_DESCRIPTION =
  "Huiyu (Yvette) Chen is a Machine Learning Engineer at Meta working on multimodal content-understanding LLMs across language, images, and video.";
export const SITE_SOCIAL_IMAGE = "/photos/profile/yvette-portrait.jpeg";
export const SITE_RSS_PATH = "/feed.xml";
export const LINKEDIN_URL =
  "https://www.linkedin.com/in/yvette-huiyu-chen";

export const RSS_ALTERNATE = {
  "application/rss+xml": SITE_RSS_PATH,
} as const;

export function absoluteUrl(pathname: string) {
  return new URL(pathname, SITE_URL).toString();
}

export function serializeJsonLd(value: unknown) {
  return JSON.stringify(value).replaceAll("<", "\\u003c");
}
