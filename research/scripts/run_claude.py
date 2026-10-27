"""Claude Opus 5.5 at low effort, asked for the brand of each test title. Writes research/results/claude.json."""

from pte.claude import EFFORT, MODEL, Claude
from pte.results import save_predictions, summary, test_set, write_result


def main() -> None:
    test = test_set()
    claude = Claude()
    preds = claude.predict(test)
    assert claude.models == {MODEL}, claude.models
    save_predictions("claude", test, preds)
    write_result("claude", {**summary(test, preds), "model": MODEL, "effort": EFFORT})


if __name__ == "__main__":
    main()
