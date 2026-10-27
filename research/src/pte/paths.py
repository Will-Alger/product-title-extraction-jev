"""Where data, caches and results live."""

from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[2]
REPO = RESEARCH.parent
RAW = REPO / "data" / "raw"
PROCESSED = REPO / "data" / "processed"
CACHE = RESEARCH / ".cache"
RESULTS = RESEARCH / "results"

# Slim copy of the Amazon metadata written by scripts/fetch_tools_meta.py (only the fields we use).
TOOLS_META_MIN = RAW / "tools_meta_min.jsonl"
