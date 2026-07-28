import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "dist/**",
    "static-export/**",
    "next-env.d.ts",
  ]),
  {
    rules: {
      // Full-document navigation keeps the prerendered GitHub Pages export
      // independent from a live React Server Components endpoint.
      "@next/next/no-html-link-for-pages": "off",
    },
  },
]);

export default eslintConfig;
