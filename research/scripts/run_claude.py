"""Claude, asked for the brand of each test title: Opus 5.5 at low effort by default, or Haiku 5.5 with --haiku.

Writes research/results/claude.json (or claude_haiku.json).
"""

import sys

from pte.claude import EFFORT, MODEL, Claude
from pte.results import save_predictions, summary, test_set, write_result

# Haiku's effort was chosen on the skipped tuning titles (see build_datasets.py), like Opus's. On them, high scored
# 92.6 F1 at $0.095 per 1,000 titles. Max scored 93.0 at $0.31, a gap within noise at three times the price.
HAIKU, HAIKU_EFFORT = "claude-haiku-5-5", "high"


def main() -> None:
    haiku = "--haiku" in sys.argv[1:]
    name, model, effort = ("claude_haiku", HAIKU, HAIKU_EFFORT) if haiku else ("claude", MODEL, EFFORT)
    test = test_set()
    claude = Claude(model, effort)
    preds = claude.predict(test)
    assert claude.models == {model}, claude.models
    save_predictions(name, test, preds)
    write_result(name, {**summary(test, preds), "model": model, "effort": effort})


if __name__ == "__main__":
    main()
