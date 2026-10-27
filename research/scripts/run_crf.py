"""The paper's method: train the CRF and score it on the 2,000 test titles.

Trains twice: on 61,374 titles (the paper's training-set size) and on the full 218k pool (the stronger baseline).
Writes research/results/crf.json.
"""

import random

from pte.crf import CRF
from pte.data import load
from pte.paths import PROCESSED
from pte.results import save_predictions, summary, test_set, write_result

PAPER_TRAIN_SIZE = 61_374


def main() -> None:
    test = test_set()
    train = load(PROCESSED / "brand_train.jsonl")
    # Must stay: the CRF cache key includes the training order, so changing this retrains the model (slow) and can
    # move its score slightly.
    random.Random(0).shuffle(train)
    result = {}
    for name, rows in (("crf", train), ("crf_61k", train[:PAPER_TRAIN_SIZE])):
        crf = CRF()
        crf.fit(rows)
        preds = crf.predict(test)
        save_predictions(name, test, preds)
        result[name] = {**summary(test, preds), "n_train": len(rows)}
    write_result("crf", result)


if __name__ == "__main__":
    main()
