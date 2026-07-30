#!/usr/bin/env node

import { copyFile, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const source = path.join(root, "node_modules", "pyodide");
const destination = path.join(root, "public", "vendor", "pyodide");
const runtimeFiles = [
  "pyodide.mjs",
  "pyodide.asm.js",
  "pyodide.asm.wasm",
  "python_stdlib.zip",
  "pyodide-lock.json",
];

await mkdir(destination, { recursive: true });
await Promise.all(
  runtimeFiles.map((filename) =>
    copyFile(path.join(source, filename), path.join(destination, filename)),
  ),
);

await writeFile(
  path.join(destination, "NOTICE.txt"),
  [
    "Pyodide 0.29.3 runtime files",
    "",
    "The files in this directory are generated from the pinned `pyodide` npm",
    "package so the interactive tutorial can run without a third-party CDN.",
    "",
    "Pyodide is licensed under MPL-2.0. Source and license:",
    "https://github.com/pyodide/pyodide",
    "",
  ].join("\n"),
);

console.log(`Synced ${runtimeFiles.length} Pyodide runtime files.`);
