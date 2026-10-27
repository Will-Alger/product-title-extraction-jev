"""Build the brand train/test split from Amazon 2023 Tools & Home Improvement metadata.

Labels follow the paper's distant supervision (section 6.2): a title is labeled only when its catalog brand
appears in it (ignoring case and punctuation), with at most 3 titles per brand, plus a small share of
unbranded titles. Titles are deduplicated, so no test title also appears in training.
"""

import json
import random
from collections import Counter, defaultdict

from pte.data import Example, save
from pte.paths import PROCESSED, TOOLS_META_MIN
from pte.text import brand_key, find_token_span, loose_key, tokenize

SEED = 13
UNBRANDED_NAMES = {"generic", "unknown", "unbranded", "nobrand", "na", "none", "oem"}
PER_BRAND_CAP = 3
UNBRANDED_SHARE = 0.04
N_TEST, N_DEV = 2000, 1000
# An earlier pass set aside 400 Power Tools / Hand Tools titles for a separate labeled set, no longer used. They stay out
# so the splits (and every cached result) are unchanged.
SET_ASIDE_CATEGORIES, N_SET_ASIDE = {"Power Tools", "Hand Tools"}, 400


def main() -> None:
    rng = random.Random(SEED)
    with TOOLS_META_MIN.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    rng.shuffle(records)

    # Canonical spelling per brand: the most common one across the catalog.
    spellings: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        b = r["details"].get("Brand") or r["store"]
        if b:
            spellings[brand_key(b)][b.strip()] += 1
    canonical = {k: c.most_common(1)[0][0] for k, c in spellings.items()}

    pool, unbranded = [], []
    per_brand = Counter()
    seen_titles = set()
    set_aside = 0
    for r in records:
        title = r["title"].strip()
        categories = r["categories"]
        # The catalog brand field, or the seller's store name when it is missing.
        brand = (r["details"].get("Brand") or r["store"] or "").strip()
        brand_k, title_k = brand_key(brand), loose_key(title)
        if not title or not brand or not brand_k or title_k in seen_titles:
            continue
        seen_titles.add(title_k)
        is_unbranded = brand_k in UNBRANDED_NAMES
        in_set_aside_category = len(categories) > 2 and categories[2] in SET_ASIDE_CATEGORIES
        if in_set_aside_category and set_aside < N_SET_ASIDE and not is_unbranded:
            set_aside += 1
            continue
        category = " > ".join(categories[1:3])
        brand_source = "src:brand_field" if r["details"].get("Brand") else "src:store"
        if is_unbranded:
            unbranded.append(Example(r["asin"], title, None, None, category, ["unbranded", brand_source]))
            continue
        span = find_token_span(tokenize(title), brand)
        if span is not None and per_brand[brand_k] < PER_BRAND_CAP:
            per_brand[brand_k] += 1
            pool.append(Example(r["asin"], title, canonical[brand_k], span, category, [brand_source]))

    pool += unbranded[: int(len(pool) * UNBRANDED_SHARE)]
    rng.shuffle(pool)
    # The N_DEV titles between test and train are skipped. An earlier version used them to choose the CRF's settings
    # and Claude's effort level (that tuning code isn't included); skipping them keeps the splits unchanged.
    test, train = pool[:N_TEST], pool[N_TEST + N_DEV :]
    save(PROCESSED / "brand_train.jsonl", train)
    save(PROCESSED / "brand_test.jsonl", test)
    print(f"train={len(train)} test={len(test)} unbranded-in-test={sum(e.value is None for e in test)}")


if __name__ == "__main__":
    main()
