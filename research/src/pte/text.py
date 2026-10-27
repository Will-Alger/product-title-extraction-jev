"""Tokenization and string keys shared by every method.

Tokens follow the paper: split on whitespace and on standalone punctuation, but keep
internal joiners so "20V", "1/2", "Drill/Driver", "B4L03A#B1H" and "J&C" stay single tokens.
"""

import re
import unicodedata
from dataclasses import dataclass

_TOKEN = re.compile(r"[\w\"'’]+(?:[&./#+\-'’][\w\"'’]+)*|[^\s\w]", re.UNICODE)
_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_CORP_WORDS = re.compile(r"\b(the|inc|incorporated|llc|ltd|limited|co|corp|corporation|company|brand|brands)\b")


@dataclass(frozen=True, slots=True)
class Token:
    text: str
    start: int
    end: int


def tokenize(title: str) -> list[Token]:
    return [Token(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(title)]


def _fold(s: str) -> str:
    """Lowercase and strip accents, so "Bollé" compares equal to "Bolle"."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()


def loose_key(s: str) -> str:
    """Loose identity key: case, accent, spacing and punctuation insensitive ("J & C" == "J&C" == "jc")."""
    return _NON_ALNUM.sub("", _fold(s))


def brand_key(s: str) -> str:
    """Like loose_key, but also drops words such as "The", "Inc" and "Company", so "THE CHENILLE KRAFT COMPANY"
    == "ChenilleKraft". Used to group catalog spellings and to score answers."""
    s = _CORP_WORDS.sub(" ", _NON_ALNUM.sub(" ", _fold(s)))
    return s.replace(" ", "")


def find_token_span(tokens: list[Token], value: str, keyfn=loose_key) -> tuple[int, int] | None:
    """Earliest run of tokens whose key equals the value's key (the paper's distant-supervision match).

    Building the data uses the default loose_key, which keeps words like "Inc": a catalog brand "Acme Inc" only
    matches a title that also says "Inc".
    """
    target = keyfn(value)
    if not target:
        return None
    n = len(tokens)
    keys = [keyfn(t.text) for t in tokens]
    for i in range(n):
        # Spans start and end on a token with content, so "," or "】" never becomes part of a label.
        if not keys[i]:
            continue
        acc = ""
        for j in range(i, n):
            acc += keys[j]
            if acc == target and keys[j]:
                return i, j + 1
            if not target.startswith(acc):
                break
    return None


def span_text(title: str, tokens: list[Token], span: tuple[int, int]) -> str:
    """The title text covered by a token span, as written."""
    return title[tokens[span[0]].start : tokens[span[1] - 1].end]


def bio(n_tokens: int, span: tuple[int, int] | None) -> list[str]:
    """Begin/inside/outside tags: B for the span's first token, I for the rest of it, O everywhere else."""
    labels = ["O"] * n_tokens
    if span:
        s, e = span
        labels[s] = "B"
        for i in range(s + 1, e):
            labels[i] = "I"
    return labels


def span_from_bio(labels: list[str]) -> tuple[int, int] | None:
    """Paper section 6.3: first B followed by its run of I's; all-O means no value."""
    for i, lab in enumerate(labels):
        if lab == "B":
            j = i + 1
            while j < len(labels) and labels[j] == "I":
                j += 1
            return i, j
    return None
