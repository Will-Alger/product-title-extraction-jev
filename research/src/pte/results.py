"""Saving and loading each method's answers, so later scripts can compare methods without rerunning them.

Answers go to data/processed/predictions/ (git-ignored, like all derived data); summaries go to research/results/.
"""

import json

from pte.data import Example, Prediction, load
from pte.metrics import f1
from pte.paths import PROCESSED, RESULTS

PREDICTIONS = PROCESSED / "predictions"


def test_set() -> list[Example]:
    return load(PROCESSED / "brand_test.jsonl")


def cost_per_1k(preds: list[Prediction]) -> float:
    return 1000 * sum(p.cost_usd for p in preds) / len(preds)


def summary(test: list[Example], preds: list[Prediction]) -> dict:
    return {"f1": f1(test, preds), "cost_per_1k_usd": cost_per_1k(preds), "n_test": len(test)}


def save_predictions(name: str, test: list[Example], preds: list[Prediction]) -> None:
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    with (PREDICTIONS / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for ex, p in zip(test, preds):
            f.write(json.dumps({"id": ex.id, "value": p.value, "confidence": p.confidence, "cost_usd": p.cost_usd},
                               ensure_ascii=False) + "\n")


def load_predictions(name: str, test: list[Example]) -> list[Prediction]:
    path = PREDICTIONS / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"No {name} answers yet: run scripts/run_{name}.py first.")
    with path.open(encoding="utf-8") as f:
        by_id = {d.pop("id"): Prediction(**d) for d in map(json.loads, f)}
    return [by_id[ex.id] for ex in test]


def write_result(name: str, result: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
