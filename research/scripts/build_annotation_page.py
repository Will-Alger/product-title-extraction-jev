"""Build the labeling page: confirm or fix the catalog brand on 200 random test titles.

An earlier version of the page asked for blind labels, without showing the catalog brand; 11 of these titles were
labeled that way. The page then switched to pre-filling the catalog brand (from the seller listing, not from any
method being compared), and the labeler confirmed or fixed it. The 11 keep their blind labels
(`blind_before_switch` in annotation_items.json).

Writes data/processed/annotation.html (git-ignored: it embeds Amazon titles) and annotation_items.json.
"""

import json
import random

from pte.data import load
from pte.paths import PROCESSED, RESEARCH
from pte.text import tokenize

SAMPLE_POOL, N_ITEMS, SEED = 300, 200, 7
# The 11 test titles labeled blind before the page switched to confirm-or-fix.
ALREADY_BLIND = {"B00BWES708", "B00N644DQ4", "B017KYOWKC", "B01KHQTUNY", "B01LWCFCQA", "B089S3HGLF", "B08C9SFFQR",
                 "B09L4471L3", "B09MR2MC8Z", "B09TT7TVHZ", "B0C2J9ZCHM"}


def main() -> None:
    test = load(PROCESSED / "brand_test.jsonl")
    pool = random.Random(SEED).sample(test, SAMPLE_POOL)  # same draw as the first version
    chosen = [e for e in pool if e.id in ALREADY_BLIND]
    chosen += [e for e in pool if e.id not in ALREADY_BLIND][: N_ITEMS - len(chosen)]
    assert len(chosen) == N_ITEMS and len(ALREADY_BLIND) == sum(e.id in ALREADY_BLIND for e in chosen)
    items, index = [], {}
    for ex in chosen:
        iid = f"t{ex.id}"
        items.append({"id": iid, "title": ex.title, "tokens": [t.text for t in tokenize(ex.title)],
                      "suggest": list(ex.span) if ex.span else None})
        index[iid] = {"set": "brand_test", "asin": ex.id, "blind_before_switch": ex.id in ALREADY_BLIND}
    template = (RESEARCH / "annotation" / "template.html").read_text(encoding="utf-8")
    page = template.replace("/*ITEMS*/[]", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
    (PROCESSED / "annotation.html").write_text(page, encoding="utf-8")
    (PROCESSED / "annotation_items.json").write_text(json.dumps(index, indent=0), encoding="utf-8")
    print(f"{len(items)} items ({len(ALREADY_BLIND)} already labeled blind) -> {PROCESSED / 'annotation.html'}")


if __name__ == "__main__":
    main()
