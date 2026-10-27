"""Jev answers every title; the ones it is least sure about go to Claude.

Exploratory: this was designed after the main comparison. A title goes to Claude when Jev's confidence is below a
cutoff, or when Jev says "no brand". The cutoff is the lowest one that keeps F1 within MAX_F1_LOSS of Claude
alone. To avoid grading the cutoff on the titles it was picked on, the titles are split in half at random: the
cutoff is picked on one half and scored on the other. This is repeated N_SPLITS times and the scores averaged.

Run scripts/run_jev.py and scripts/run_claude.py first. Writes research/results/jev_then_claude.json.
"""

import numpy as np

from pte.metrics import f1
from pte.results import load_predictions, test_set, write_result

MAX_F1_LOSS = 0.005  # half an F1 point
N_SPLITS = 200
NO_BRAND = -1.0  # confidence given to Jev's "no brand" answers, below any real confidence, so they go to Claude


def main() -> None:
    test = test_set()
    jev, claude = load_predictions("jev", test), load_predictions("claude", test)
    conf = np.array([p.confidence if p.value is not None else NO_BRAND for p in jev])
    # Candidate cutoffs from lowest to highest: "send nothing", then each distinct Jev confidence.
    cutoffs = np.r_[-np.inf, np.unique(conf)]

    def combined(titles, cutoff):
        """F1, share sent to Claude, and $ per 1k titles when titles with confidence < cutoff go to Claude."""
        to_claude = conf[titles] < cutoff
        preds = [claude[i] if sent else jev[i] for i, sent in zip(titles, to_claude)]
        # Jev runs on every title; Claude only on the ones sent to it.
        cost = sum(jev[i].cost_usd for i in titles) + sum(claude[i].cost_usd for i in titles[to_claude])
        return f1([test[i] for i in titles], preds), to_claude.mean(), 1000 * cost / len(titles)

    def pick_cutoff(titles):
        target = f1([test[i] for i in titles], [claude[i] for i in titles]) - MAX_F1_LOSS
        # If no cutoff gets close enough, fall back to the highest one (nearly everything goes to Claude).
        return next((c for c in cutoffs if combined(titles, c)[0] >= target), cutoffs[-1])

    rng = np.random.default_rng(0)
    scores = []
    for _ in range(N_SPLITS):
        shuffled = rng.permutation(len(test))
        pick_on, score_on = shuffled[: len(test) // 2], shuffled[len(test) // 2 :]
        scores.append(combined(score_on, pick_cutoff(pick_on)))
    f1s, shares, costs = np.array(scores).T

    write_result("jev_then_claude", {
        "f1": float(f1s.mean()),
        "f1_middle_95pct_of_splits": [float(x) for x in np.percentile(f1s, [2.5, 97.5])],
        "share_sent_to_claude": float(shares.mean()),
        "cost_per_1k_usd": float(costs.mean()),
        "max_f1_loss": MAX_F1_LOSS,
        "n_splits": N_SPLITS,
    })


if __name__ == "__main__":
    main()
