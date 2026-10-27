"""Jev: one multiple-choice question per title. The options are every phrase of up to 5 words in the title,
plus NONE, and Jev picks the brand. No training data is used."""

import asyncio
import hashlib
import json
import sqlite3

from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient, Choice, RetryPolicy

from pte.data import Example, Prediction
from pte.paths import CACHE, REPO
from pte.text import Token

MODEL = "jev-latest"
USD_PER_INPUT_TOKEN = 0.042 / 1_000_000  # published price; output tokens are free
MAX_OPTIONS = 255  # a Choice takes at most 255 options
NONE = "NONE"
QUESTION = (
    "Which part of this product title is the product's brand name (the company or brand that makes and sells it)? "
    "Ignore brands only mentioned for compatibility, like 'fits X' or 'for X'. Choose NONE if the title names no brand."
)


def candidates(title: str, toks: list[Token], max_len: int = 5) -> list[str]:
    """Distinct phrases of up to max_len tokens, shortest first, that start and end on a word (not punctuation)."""
    out: dict[str, None] = {}
    for length in range(1, max_len + 1):
        for s in range(len(toks) - length + 1):
            first, last = toks[s], toks[s + length - 1]
            if not (any(c.isalnum() for c in first.text) and any(c.isalnum() for c in last.text)):
                continue
            out.setdefault(title[first.start : last.end])
            if len(out) >= MAX_OPTIONS - 1:  # leave room for NONE
                return list(out)
    return list(out)


def question(ex: Example) -> tuple[dict, dict]:
    """The (state, questions) arguments for Jev's system_one call.

    The question text, the candidate order and the option descriptions are all part of the cache key: changing
    any of them re-asks (and pays for) every title.
    """
    criteria = {text: None for text in candidates(ex.title, ex.tokens)}
    criteria[NONE] = "None of these"
    return {"product_title": ex.title}, {"value": Choice(instructions=QUESTION, criteria=criteria)}


class Jev:
    def __init__(self, concurrency: int = 16):
        load_dotenv(REPO / ".env")
        self.concurrency = concurrency
        self.models: set[str] = set()  # model versions that answered; a run should see exactly one
        # Every response is cached, so reruns are free and give the same answers.
        CACHE.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(CACHE / "jev.sqlite", timeout=120)
        self.db.execute("create table if not exists r (k text primary key, v text)")

    async def _ask(self, client, sem, state, questions) -> dict:
        # Cache key. "v": 2 marks an earlier cache-format change; editing this line re-asks every title.
        payload = {"v": 2, "m": MODEL, "s": state, "q": {k: q.model_dump() for k, q in questions.items()}}
        k = hashlib.sha256(json.dumps(payload).encode()).hexdigest()
        row = self.db.execute("select v from r where k=?", (k,)).fetchone()
        if row:
            return json.loads(row[0])
        async with sem:
            resp = await client.system_one(state=state, questions=questions, model=MODEL)
        out = {"answers": {name: a.model_dump() for name, a in resp.answers.items()},
               "input_tokens": resp.usage.input_tokens or 0,  # missing usage would count as $0
               "model": resp.model}
        self.db.execute("insert or replace into r values (?, ?)", (k, json.dumps(out)))
        self.db.commit()
        return out

    def predict(self, examples: list[Example]) -> list[Prediction]:
        async def go():
            sem = asyncio.Semaphore(self.concurrency)
            async with AsyncTypeSafeClient(retry=RetryPolicy(max_retries=6), timeout=60) as client:
                return await asyncio.gather(*(self._ask(client, sem, *question(ex)) for ex in examples))

        out = []
        for r in asyncio.run(go()):
            self.models.add(r["model"])
            a = r["answers"]["value"]
            choice = None if a["choice"] == NONE else a["choice"]
            out.append(Prediction(choice, a["probabilities"][a["choice"]], r["input_tokens"] * USD_PER_INPUT_TOKEN))
        return out
