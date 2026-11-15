"""Where do the methods differ? F1 on two slices of the branded test titles, plus a first-word baseline.

- Brand not the first word of the title, where position doesn't give the answer away.
- Brands that never appear in the 218k-title CRF's training titles.

Run scripts/run_crf.py, run_jev.py and run_claude.py (with and without --haiku) first. Writes research/results/slices.json.
"""

from pte.data import Prediction, load
from pte.metrics import f1, paired_bootstrap
from pte.paths import PROCESSED
from pte.results import load_predictions, test_set, write_result
from pte.text import brand_key, span_text


def main() -> None:
    test = test_set()
    seen = {brand_key(e.value) for e in load(PROCESSED / "brand_train.jsonl") if e.value}
    methods = {name: load_predictions(name, test) for name in ("crf", "crf_61k", "jev", "claude", "claude_haiku")}
    methods["first_word"] = [Prediction(span_text(ex.title, ex.tokens, (0, 1)) if ex.tokens else None)
                             for ex in test]

    branded = [i for i, ex in enumerate(test) if ex.value]
    slices = {
        "all": list(range(len(test))),
        "brand_not_first_word": [i for i in branded if test[i].span and test[i].span[0] != 0],
        "seen_brand": [i for i in branded if brand_key(test[i].value) in seen],
        "unseen_brand": [i for i in branded if brand_key(test[i].value) not in seen],
    }

    result = {}
    for name, idx in slices.items():
        golds = [test[i] for i in idx]
        row = {"n": len(idx)}
        for m, preds in methods.items():
            row[m] = f1(golds, [preds[i] for i in idx])
        # Uncorrected 95% intervals, for scale only: these splits were planned as descriptive, not as tests.
        for other in ("crf", "crf_61k", "claude_haiku"):
            diff, lo, hi = paired_bootstrap(golds, [methods["jev"][i] for i in idx], [methods[other][i] for i in idx])
            row[f"jev_minus_{other}"] = {"diff": diff, "ci95": [lo, hi]}
        result[name] = row
    write_result("slices", result)


if __name__ == "__main__":
    main()
