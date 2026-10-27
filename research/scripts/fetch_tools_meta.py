"""Stream Amazon Reviews 2023 Tools & Home Improvement metadata and keep only the fields we use.

The source file is ~4.85 GB; only a slim JSONL (title, brand, categories, a few detail fields)
is written to data/raw/tools_meta_min.jsonl.
"""

import json
import sys

import httpx

from pte.paths import RAW, TOOLS_META_MIN

URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
    "raw/meta_categories/meta_Tools_and_Home_Improvement.jsonl"
)
DETAIL_KEYS = ("Brand", "Manufacturer", "Color", "Voltage", "Size", "Material", "Item model number", "Power Source")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    tmp = TOOLS_META_MIN.with_suffix(".partial")
    seen = kept = 0
    with httpx.stream("GET", URL, follow_redirects=True, timeout=120) as resp, tmp.open("w", encoding="utf-8") as out:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            seen += 1
            rec = json.loads(line)
            title = rec.get("title")
            if not title:
                continue
            details = rec.get("details") or {}
            slim = {
                "asin": rec["parent_asin"],
                "title": title,
                "store": rec.get("store"),
                "categories": rec.get("categories") or [],
                "details": {k: details[k] for k in DETAIL_KEYS if isinstance(details.get(k), str)},
            }
            out.write(json.dumps(slim, ensure_ascii=False) + "\n")
            kept += 1
            if seen % 50_000 == 0:
                print(f"{seen:,} read, {kept:,} kept", flush=True)
    tmp.replace(TOOLS_META_MIN)
    print(f"done: {seen:,} read, {kept:,} kept -> {TOOLS_META_MIN}", file=sys.stderr)


if __name__ == "__main__":
    main()
