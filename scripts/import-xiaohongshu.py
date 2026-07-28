#!/usr/bin/env python3
"""Import the public profile summary and visible note cards from Xiaohongshu."""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/tmp/xhs-link.html")
OUTPUT = ROOT / "content" / "xiaohongshu.json"
IMAGE_DIR = ROOT / "public" / "xiaohongshu"
PROFILE_URL = "https://xhslink.cn/m/3SSS5q3QCtU"


def extract_state(page: str):
    match = re.search(
        r"window\.__INITIAL_STATE__=(\{.*?\})</script>",
        page,
        flags=re.S,
    )
    if not match:
        raise RuntimeError("Could not find Xiaohongshu profile state.")
    return json.loads(match.group(1).replace("undefined", "null"))


def download(url: str, destination: Path):
    if url.startswith("//"):
        url = f"https:{url}"
    request = urllib.request.Request(
        url,
        headers={
            "Referer": "https://www.xiaohongshu.com/",
            "User-Agent": (
                "Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) "
                "AppleWebKit/605.1.15 Mobile/15E148"
            ),
        },
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        destination.write_bytes(response.read())


def main():
    page = SOURCE.read_text()
    state = extract_state(page)
    profile = state["profile"]["userInfo"]
    notes = state["profile"]["noteData"]
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    imported_notes = []
    for index, note in enumerate(notes):
        image_path = IMAGE_DIR / f"note-{index + 1}.jpg"
        download(note["cover"]["url"], image_path)
        imported_notes.append(
            {
                "id": note["id"],
                "title": note["title"].strip(),
                "likes": note["likes"],
                "comments": note["comments"],
                "collects": note["collects"],
                "image": f"/xiaohongshu/{image_path.name}",
                "sticky": note["sticky"],
            }
        )

    output = {
        "nickname": profile["nickname"],
        "description": profile["desc"],
        "location": profile["ipLocation"],
        "followers": profile["fans"],
        "following": profile["follows"],
        "likesAndCollects": profile["likeAndCollect"],
        "redId": profile["redId"],
        "profileUrl": PROFILE_URL,
        "notes": imported_notes,
    }
    OUTPUT.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Imported {len(imported_notes)} public notes into {OUTPUT}")


if __name__ == "__main__":
    main()
