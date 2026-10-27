"""Time Jev and Claude on 200 test titles, one request at a time, with no cache. Writes research/results/latency.json."""

import asyncio
import json
import random
import time

import anthropic
import numpy as np
from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient

from pte import claude, jev
from pte.data import load
from pte.paths import PROCESSED, REPO, RESULTS

N = 200


def summarize(ms: list[float]) -> dict:
    return {"n": len(ms), "p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)),
            "mean_ms": float(np.mean(ms)), "measured_at": time.strftime("%Y-%m-%d %H:%M")}


async def time_jev(exs) -> list[float]:
    out = []
    async with AsyncTypeSafeClient(timeout=60) as client:
        for ex in exs:
            state, questions = jev.question(ex)
            t0 = time.perf_counter()
            await client.system_one(state=state, questions=questions, model=jev.MODEL)
            out.append((time.perf_counter() - t0) * 1000)
    return out


async def time_claude(exs) -> list[float]:
    out = []
    async with anthropic.AsyncAnthropic() as client:
        for ex in exs:
            t0 = time.perf_counter()
            await client.messages.parse(**claude.request_kwargs(ex))
            out.append((time.perf_counter() - t0) * 1000)
    return out


def main() -> None:
    load_dotenv(REPO / ".env")
    exs = random.Random(11).sample(load(PROCESSED / "brand_test.jsonl"), N)
    res = {"jev": summarize(asyncio.run(time_jev(exs))), "claude": summarize(asyncio.run(time_claude(exs))),
           "note": "sequential, uncached, one Windows laptop on a residential network"}
    (RESULTS / "latency.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
