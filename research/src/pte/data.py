"""A labeled title (the catalog brand and where it sits in the title), and a method's answer for one title."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pte.text import Token, bio, brand_key, tokenize


@dataclass
class Example:
    id: str
    title: str
    # Catalog brand, or None when the product is unbranded.
    value: str | None
    # Token span of the brand in the title (half-open), or None.
    span: tuple[int, int] | None
    # Kept for inspecting the data; no method or result uses them.
    category: str = ""
    tags: list[str] = field(default_factory=list)

    def accepts(self, pred: str) -> bool:
        return self.value is not None and brand_key(pred) == brand_key(self.value)

    @property
    def tokens(self) -> list[Token]:
        return tokenize(self.title)

    @property
    def labels(self) -> list[str]:
        return bio(len(self.tokens), self.span)


@dataclass
class Prediction:
    # Predicted brand, or None for "no brand".
    value: str | None
    # The method's confidence in its answer, when it gives one.
    confidence: float | None = None
    cost_usd: float = 0.0


def save(path: Path, examples: list[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")


def load(path: Path) -> list[Example]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            d["span"] = tuple(d["span"]) if d["span"] is not None else None
            out.append(Example(**d))
    return out
