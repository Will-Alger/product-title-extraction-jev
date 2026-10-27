"""How often is the catalog brand (the answer key) wrong? Compares it with one person's labels on 200 test titles.

Build the page with scripts/build_annotation_page.py, label, then export the labels to
data/processed/human_labels/labels/<annotator>/items/<item>.json. The labels themselves aren't in the repo.

Most titles were labeled by confirming or fixing the pre-filled catalog brand, which may bias the check toward
agreeing with the catalog; 11 were labeled blind. "Can't tell" titles are left out. Writes
research/results/human_check.json.
"""

import json
import math
from collections import Counter

from pte.data import Example, load
from pte.paths import PROCESSED, RESULTS

LABELS = PROCESSED / "human_labels" / "labels"


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """95% interval for a proportion k/n."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def agrees(catalog: Example, human: str | None) -> bool:
    """Human None means "no brand"."""
    return catalog.value is None if human is None else catalog.accepts(human)


def main() -> None:
    index = json.loads((PROCESSED / "annotation_items.json").read_text())
    test = {e.id: e for e in load(PROCESSED / "brand_test.jsonl")}
    paths = [p for p in LABELS.glob("*/items/*.json") if p.stem in index]
    labels = {}
    for path in paths:
        doc = json.loads(path.read_text(encoding="utf-8"))
        # The page's export wraps each label in "data". With one annotator there is one file per title.
        labels.setdefault(path.stem, doc.get("data", doc))

    how_labeled, wrong = Counter(), []
    n_cant_tell = n_judged = 0
    for item, meta in sorted(index.items()):
        lab = labels.get(item)
        if lab is None:
            how_labeled["missing"] += 1
            continue
        how_labeled["blind" if meta["blind_before_switch"] else lab.get("method", "unknown")] += 1
        if lab["brandStatus"] == "unsure":
            n_cant_tell += 1
            continue
        n_judged += 1
        catalog = test[meta["asin"]]
        human = lab["brand"]["text"] if lab["brandStatus"] == "span" else None
        if not agrees(catalog, human):
            wrong.append({"catalog": catalog.value, "human": human})

    summary = {
        "annotators": len({p.parent.parent.name for p in paths}),
        "items": len(index),
        "how_labeled": dict(how_labeled),
        "n_cant_tell": n_cant_tell,
        "n_judged": n_judged,
        "catalog_error_rate": len(wrong) / n_judged,
        "catalog_error_wilson95": wilson(len(wrong), n_judged),
        "catalog_errors_by_kind": dict(Counter(
            "catalog brand, human none" if w["human"] is None else
            "catalog none, human brand" if w["catalog"] is None else "different brand" for w in wrong)),
    }
    (RESULTS / "human_check.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
