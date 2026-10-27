"""Jev with no training data: one multiple-choice question per test title. Then compare it with the CRF.

Run scripts/run_crf.py first. Writes research/results/jev.json.
"""

from pte.jev import Jev
from pte.metrics import paired_bootstrap
from pte.results import load_predictions, save_predictions, summary, test_set, write_result


def main() -> None:
    test = test_set()
    jev = Jev()
    preds = jev.predict(test)
    # "jev-latest" resolves to a specific version; every answer should come from the same one.
    assert len(jev.models) == 1, jev.models
    save_predictions("jev", test, preds)

    # Is Jev's gap to the CRF bigger than chance? Resample the same titles for both, 5,000 times.
    diff, lo, hi = paired_bootstrap(test, preds, load_predictions("crf", test))
    write_result("jev", {**summary(test, preds), "model": jev.models.pop(),
                         "f1_minus_crf": {"diff": diff, "ci95": [lo, hi]}})


if __name__ == "__main__":
    main()
