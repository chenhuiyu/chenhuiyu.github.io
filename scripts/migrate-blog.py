#!/usr/bin/env python3
"""Extract the rendered Hexo posts into a portable, sanitized JSON bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlparse

from lxml import html


LOCALE_SUFFIX = re.compile(r"\.(en|zh-CN)$")
UNSAFE_TAGS = {"script", "style", "iframe", "object", "embed", "form"}


def clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    value = re.sub(r"[^\w\u3400-\u9fff]+", "-", value, flags=re.UNICODE)
    return value.strip("-")[:92]


def normalize_old_path(value: str) -> str:
    parsed = urlparse(value)
    path = unquote(parsed.path).strip("/")
    return re.sub(r"/+", "/", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()

    candidates = sorted(source.glob("20*/**/index.html"))
    records: list[dict[str, object]] = []
    path_to_slug: dict[str, str] = {}
    pair_map: dict[str, list[dict[str, object]]] = {}

    for file_path in candidates:
        relative_dir = file_path.parent.relative_to(source).as_posix()
        parts = relative_dir.split("/")
        if len(parts) < 5:
            continue

        basename = parts[-1]
        locale_match = LOCALE_SUFFIX.search(basename)
        if not locale_match:
            continue

        language = locale_match.group(1)
        base_name = LOCALE_SUFFIX.sub("", basename)
        date = "-".join(parts[:3])
        category = parts[3]
        pair_key = "/".join(parts[:4] + [base_name])
        short_hash = hashlib.sha1(relative_dir.encode("utf-8")).hexdigest()[:7]
        slug = f"{date}-{slugify(base_name)}-{language.lower()}-{short_hash}"

        document = html.fromstring(file_path.read_text(encoding="utf-8"))
        title_nodes = document.xpath(
            '//h1[contains(concat(" ", normalize-space(@class), " "), " post-title ")]'
        )
        title = clean_text(title_nodes[0].text_content() if title_nodes else "")
        created_nodes = document.xpath(
            '//time[contains(concat(" ", normalize-space(@class), " "), " post-meta-date-created ")]'
        )
        updated_nodes = document.xpath(
            '//time[contains(concat(" ", normalize-space(@class), " "), " post-meta-date-updated ")]'
        )
        created = (
            created_nodes[0].get("datetime", "")[:10]
            if created_nodes
            else date
        )
        updated = (
            updated_nodes[0].get("datetime", "")[:10]
            if updated_nodes
            else created
        )
        tag_nodes = document.xpath(
            '//a[contains(concat(" ", normalize-space(@class), " "), " post-meta__tags ")]'
        )
        tags = [clean_text(node.text_content()) for node in tag_nodes]

        record: dict[str, object] = {
            "slug": slug,
            "pairKey": pair_key,
            "title": title or base_name.replace("-", " "),
            "date": created,
            "updated": updated,
            "category": category,
            "language": language,
            "tags": tags,
            "oldPath": "/" + relative_dir + "/",
            "content": "",
            "excerpt": "",
            "toc": [],
            "alternateSlug": None,
        }
        records.append(record)
        path_to_slug[normalize_old_path("/" + relative_dir + "/")] = slug
        pair_map.setdefault(pair_key, []).append(record)

    for pair in pair_map.values():
        if len(pair) == 2:
            pair[0]["alternateSlug"] = pair[1]["slug"]
            pair[1]["alternateSlug"] = pair[0]["slug"]

    record_by_old_path = {
        normalize_old_path(str(record["oldPath"])): record for record in records
    }

    for file_path in candidates:
        relative_dir = file_path.parent.relative_to(source).as_posix()
        record = record_by_old_path.get(normalize_old_path(relative_dir))
        if not record:
            continue

        document = html.fromstring(file_path.read_text(encoding="utf-8"))
        article_nodes = document.xpath('//article[@id="article-container"]')
        if not article_nodes:
            continue
        article = article_nodes[0]

        for node in list(article.iter()):
            if node.tag in UNSAFE_TAGS:
                node.drop_tree()
                continue

            for attr_name in list(node.attrib):
                attr_value = node.attrib.get(attr_name, "")
                if attr_name.lower().startswith("on"):
                    del node.attrib[attr_name]
                elif attr_name == "style":
                    del node.attrib[attr_name]
                elif attr_name in {"href", "src"} and re.match(
                    r"^\s*javascript:", attr_value, re.I
                ):
                    del node.attrib[attr_name]

        first_heading = article.xpath(".//h1")
        if first_heading and clean_text(first_heading[0].text_content()) == record["title"]:
            first_heading[0].drop_tree()

        for anchor in article.xpath(
            './/a[contains(concat(" ", normalize-space(@class), " "), " headerlink ")]'
        ):
            anchor.drop_tree()

        for anchor in article.xpath(".//a[@href]"):
            href = anchor.get("href", "")
            normalized = normalize_old_path(href)
            if normalized in path_to_slug:
                fragment = urlparse(href).fragment
                new_href = f"/blog/{path_to_slug[normalized]}"
                if fragment:
                    new_href += f"#{fragment}"
                anchor.set("href", new_href)
            elif href.startswith("https://chenhuiyu.github.io/"):
                normalized = normalize_old_path(href)
                if normalized in path_to_slug:
                    anchor.set("href", f"/blog/{path_to_slug[normalized]}")
            elif href.startswith(("http://", "https://")):
                anchor.set("target", "_blank")
                anchor.set("rel", "noreferrer noopener")

        for image in article.xpath(".//img"):
            image.set("loading", "lazy")
            image.set("decoding", "async")
            src = image.get("src", "")
            if src.startswith("//"):
                image.set("src", "https:" + src)

        headings: list[dict[str, str]] = []
        for heading in article.xpath(".//h2 | .//h3"):
            heading_id = heading.get("id")
            heading_text = clean_text(heading.text_content())
            if heading_id and heading_text:
                headings.append(
                    {
                        "id": heading_id,
                        "text": heading_text,
                        "level": heading.tag,
                    }
                )

        paragraphs = article.xpath(".//p")
        excerpt = ""
        for paragraph in paragraphs:
            candidate = clean_text(paragraph.text_content())
            if len(candidate) > 38:
                excerpt = candidate[:220].rstrip() + ("…" if len(candidate) > 220 else "")
                break

        record["content"] = "".join(
            html.tostring(child, encoding="unicode", method="html")
            for child in article
        )
        record["excerpt"] = excerpt
        record["toc"] = headings

    records.sort(
        key=lambda item: (str(item["date"]), str(item["language"])),
        reverse=True,
    )

    output_dir = destination / "content"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "posts.json"
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    source_images = source / "img"
    destination_images = destination / "public" / "img"
    if source_images.exists():
        shutil.copytree(source_images, destination_images, dirs_exist_ok=True)

    print(
        json.dumps(
            {
                "posts": len(records),
                "languages": {
                    language: sum(
                        1 for record in records if record["language"] == language
                    )
                    for language in ("en", "zh-CN")
                },
                "categories": sorted({str(record["category"]) for record in records}),
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
