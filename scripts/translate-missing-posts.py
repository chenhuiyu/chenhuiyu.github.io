#!/usr/bin/env python3
"""Replace duplicated pseudo-bilingual posts with real translations.

The migrated Hexo archive contains 51 bilingual pairs. Some pairs point to two
different paths but contain byte-for-byte equivalent article bodies. This
script detects those pairs, translates only the missing language, and keeps the
original HTML structure, code blocks, formulae, links, images, and heading IDs.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from lxml import html


ROOT = Path(__file__).resolve().parents[1]
POSTS_PATH = ROOT / "content" / "posts.json"
TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"
MARKER_RE = re.compile(r"\[\[\[T(\d{6})\]\]\]")
CJK_RE = re.compile(r"[\u3400-\u9fff]")
LETTER_RE = re.compile(r"[A-Za-z]")
SKIP_TAGS = {"code", "pre", "kbd", "samp", "script", "style", "math"}


def plain_text(content: str) -> str:
    fragment = html.fragment_fromstring(f"<div>{content}</div>")
    return " ".join(fragment.text_content().split())


def cjk_ratio(text: str) -> float:
    cjk = len(CJK_RE.findall(text))
    latin = len(LETTER_RE.findall(text))
    return cjk / max(1, cjk + latin)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_translatable(text: str, source_language: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if source_language == "zh-CN":
        return bool(CJK_RE.search(stripped))
    if len(stripped) < 2:
        return False
    if not LETTER_RE.search(stripped):
        return False
    if stripped.startswith(("$", "\\(", "\\[", "{", "[")):
        return False
    symbols = sum(not char.isalnum() and not char.isspace() for char in stripped)
    return symbols / max(1, len(stripped)) < 0.45


def translation_text(payload: Any) -> str:
    return "".join(piece[0] for piece in payload[0] if piece and piece[0])


def request_translation(text: str, source: str, target: str) -> str:
    data = {
        "client": "gtx",
        "sl": source,
        "tl": target,
        "dt": "t",
        "q": text,
    }
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(
                TRANSLATE_URL,
                data=urllib.parse.urlencode(data).encode(),
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(request, timeout=75) as response:
                return translation_text(json.loads(response.read()))
        except Exception as error:  # pragma: no cover - network recovery
            last_error = error
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Translation failed after retries: {last_error}")


def make_batches(items: list[tuple[int, str]], max_chars: int = 3800):
    batches: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    current_size = 0
    for item_id, text in items:
        marked_size = len(text) + 18
        if current and current_size + marked_size > max_chars:
            batches.append(current)
            current = []
            current_size = 0
        current.append((item_id, text))
        current_size += marked_size
    if current:
        batches.append(current)
    return batches


def translate_batch(
    batch: list[tuple[int, str]], source: str, target: str
) -> dict[int, str]:
    payload = "\n".join(f"[[[T{item_id:06d}]]]\n{text}" for item_id, text in batch)
    translated = request_translation(payload, source, target)
    matches = list(MARKER_RE.finditer(translated))
    if len(matches) != len(batch):
        raise RuntimeError(
            f"Expected {len(batch)} translation markers, received {len(matches)}"
        )
    output: dict[int, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(translated)
        output[int(match.group(1))] = translated[match.end() : end].strip()
    return output


def translate_items(
    items: list[tuple[int, str]], source: str, target: str
) -> dict[int, str]:
    batches = make_batches(items)
    translated: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(translate_batch, batch, source, target)
            for batch in batches
        ]
        for future in as_completed(futures):
            translated.update(future.result())
    return translated


def collect_slots(root: html.HtmlElement, source_language: str):
    slots: list[tuple[html.HtmlElement, str, str]] = []
    for element in root.iter():
        inside_skipped = element.tag in SKIP_TAGS or any(
            ancestor.tag in SKIP_TAGS for ancestor in element.iterancestors()
        )
        if (
            not inside_skipped
            and element.text
            and is_translatable(element.text, source_language)
        ):
            slots.append((element, "text", element.text))
        if element.tail and is_translatable(element.tail, source_language):
            parent = element.getparent()
            parent_is_skipped = parent is None or parent.tag in SKIP_TAGS or any(
                ancestor.tag in SKIP_TAGS for ancestor in parent.iterancestors()
            )
            if not parent_is_skipped:
                slots.append((element, "tail", element.tail))
    return slots


def preserve_outer_whitespace(original: str, translated: str) -> str:
    leading = re.match(r"^\s*", original).group(0)
    trailing = re.search(r"\s*$", original).group(0)
    return f"{leading}{translated.strip()}{trailing}"


def translate_post(
    source_post: dict[str, Any],
    target_post: dict[str, Any],
    source_language: str,
    target_language: str,
):
    root = html.fragment_fromstring(f"<div>{source_post['content']}</div>")
    slots = collect_slots(root, source_language)

    metadata = [
        source_post["title"],
        source_post["excerpt"],
        *[item["text"] for item in source_post["toc"]],
    ]
    all_items = [(index, value) for index, value in enumerate(metadata)]
    slot_offset = len(all_items)
    all_items.extend(
        (slot_offset + index, original)
        for index, (_, _, original) in enumerate(slots)
    )

    translated = translate_items(all_items, source_language, target_language)
    target_post["title"] = translated[0]
    target_post["excerpt"] = translated[1]

    for index, item in enumerate(target_post["toc"]):
        item["text"] = translated[2 + index]

    for index, (element, attribute, original) in enumerate(slots):
        value = preserve_outer_whitespace(
            original, translated[slot_offset + index]
        )
        setattr(element, attribute, value)

    target_post["content"] = "".join(
        html.tostring(child, encoding="unicode", method="html") for child in root
    )


def clean_remaining_chinese(post: dict[str, Any]) -> bool:
    """Translate Chinese fragments left in an otherwise English article."""
    root = html.fragment_fromstring(f"<div>{post['content']}</div>")
    slots = collect_slots(root, "zh-CN")
    metadata_slots: list[tuple[str, int | None, str]] = []
    if CJK_RE.search(post["title"]):
        metadata_slots.append(("title", None, post["title"]))
    if CJK_RE.search(post["excerpt"]):
        metadata_slots.append(("excerpt", None, post["excerpt"]))
    for index, item in enumerate(post["toc"]):
        if CJK_RE.search(item["text"]):
            metadata_slots.append(("toc", index, item["text"]))

    if not slots and not metadata_slots:
        return False

    items = [
        (index, value) for index, (_, _, value) in enumerate(metadata_slots)
    ]
    offset = len(items)
    items.extend(
        (offset + index, original)
        for index, (_, _, original) in enumerate(slots)
    )
    translated = translate_items(items, "zh-CN", "en")

    for index, (kind, toc_index, _) in enumerate(metadata_slots):
        if kind == "title":
            post["title"] = translated[index]
        elif kind == "excerpt":
            post["excerpt"] = translated[index]
        elif toc_index is not None:
            post["toc"][toc_index]["text"] = translated[index]

    for index, (element, attribute, original) in enumerate(slots):
        setattr(
            element,
            attribute,
            preserve_outer_whitespace(original, translated[offset + index]),
        )
    post["content"] = "".join(
        html.tostring(child, encoding="unicode", method="html") for child in root
    )
    return True


def main():
    posts = json.loads(POSTS_PATH.read_text())
    by_slug = {post["slug"]: post for post in posts}
    handled: set[str] = set()
    duplicate_pairs = []

    for post in posts:
        if post["pairKey"] in handled:
            continue
        handled.add(post["pairKey"])
        alternate = by_slug[post["alternateSlug"]]
        post_text = normalize(plain_text(post["content"]))
        alternate_text = normalize(plain_text(alternate["content"]))
        if post_text != alternate_text:
            continue
        duplicate_pairs.append((post, alternate))

    print(f"Found {len(duplicate_pairs)} duplicated bilingual pairs.")
    for index, (first, second) in enumerate(duplicate_pairs, start=1):
        actual_language = (
            "zh-CN" if cjk_ratio(plain_text(first["content"])) >= 0.12 else "en"
        )
        target_language = "en" if actual_language == "zh-CN" else "zh-CN"
        source = first if first["language"] == actual_language else second
        target = first if first["language"] == target_language else second

        print(
            f"[{index:02d}/{len(duplicate_pairs)}] "
            f"{actual_language} → {target_language}: {source['title']}"
        )
        translate_post(source, target, actual_language, target_language)

    cleanup_count = 0
    for pass_number in range(2):
        for post in posts:
            if post["language"] == "en" and clean_remaining_chinese(post):
                cleanup_count += 1
        print(f"Completed English cleanup pass {pass_number + 1}.")

    for post in posts:
        if (
            post["language"] == "zh-CN"
            and post["pairKey"].endswith(
                "evaluation-of-generation-based-large-language-models-llms-opportunities-and-challenges-from-generation-to-judgment"
            )
        ):
            post["title"] = (
                "基于生成的大语言模型（LLM）评估："
                "从生成到判断的机遇与挑战"
            )

    POSTS_PATH.write_text(
        json.dumps(posts, ensure_ascii=False, separators=(",", ":"))
    )
    print(f"Updated {POSTS_PATH}; cleaned {cleanup_count} English articles.")


if __name__ == "__main__":
    main()
