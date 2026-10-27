"""Claude reads the title and writes the brand as free text. Each response records the model that answered, and
scripts/run_claude.py checks it is always MODEL."""

import asyncio
import hashlib
import json
import sqlite3

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

from pte.data import Example, Prediction
from pte.paths import CACHE, REPO

MODEL = "claude-opus-5-5"
EFFORT = "low"
USD_IN, USD_OUT = 4.00 / 1_000_000, 20.00 / 1_000_000  # list prices per token used for these runs
# Claude is also asked for a confidence; no result uses it, but it is part of the cached prompt, so it stays.
SYSTEM = (
    "You extract product attributes from e-commerce product titles. Return the brand exactly as it is written in "
    "the title, or null when the title names no brand of its own. Brands mentioned only for compatibility "
    "('fits X', 'for X') are not the product's brand. Also give your confidence (0 to 1) that the answer is right."
)


class BrandAnswer(BaseModel):
    brand: str | None
    confidence: float


def user_message(ex: Example) -> str:
    return f"Product title: {ex.title}"


def request_kwargs(ex: Example) -> dict:
    """The messages.parse arguments for one title (shared with the timing script)."""
    return dict(model=MODEL, max_tokens=4096, output_config={"effort": EFFORT}, system=SYSTEM,
                messages=[{"role": "user", "content": user_message(ex)}], output_format=BrandAnswer)


class Claude:
    def __init__(self, concurrency: int = 8):
        load_dotenv(REPO / ".env")
        self.concurrency = concurrency
        self.models: set[str] = set()  # models that answered; should be exactly {MODEL}
        # Every response is cached, so reruns are free and give the same answers.
        CACHE.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(CACHE / "claude.sqlite", timeout=120)
        self.db.execute("create table if not exists r (k text primary key, v text)")

    async def _ask(self, client, sem, ex: Example) -> dict:
        # Cache key. "v2" marks an earlier cache-format change; editing this line re-asks every title.
        k = hashlib.sha256(json.dumps(["v2", MODEL, EFFORT, SYSTEM, user_message(ex)]).encode()).hexdigest()
        row = self.db.execute("select v from r where k=?", (k,)).fetchone()
        if row:
            return json.loads(row[0])
        async with sem:
            resp = await client.messages.parse(**request_kwargs(ex))
        refused = resp.stop_reason == "refusal"
        out = {
            "parsed": None if refused or resp.parsed_output is None else resp.parsed_output.model_dump(),
            "answered_by": resp.model,
            # Priced when the call is made and cached with the response.
            "cost": resp.usage.input_tokens * USD_IN + resp.usage.output_tokens * USD_OUT,
        }
        self.db.execute("insert or replace into r values (?, ?)", (k, json.dumps(out)))
        self.db.commit()
        return out

    def predict(self, examples: list[Example]) -> list[Prediction]:
        async def go():
            sem = asyncio.Semaphore(self.concurrency)
            async with anthropic.AsyncAnthropic(max_retries=6) as client:
                return await asyncio.gather(*(self._ask(client, sem, ex) for ex in examples))

        out = []
        for r in asyncio.run(go()):
            self.models.add(r["answered_by"])
            p = r["parsed"] or {"brand": None, "confidence": None}
            # An empty string, like null, means "no brand".
            out.append(Prediction(p["brand"] or None, p["confidence"], r["cost"]))
        return out
