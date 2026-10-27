"""The paper's method: a linear-chain CRF over begin/inside/outside tags with its feature set (section 7.2)."""

import hashlib
import pickle
from collections import Counter, defaultdict

import sklearn_crfsuite

from pte.data import Example, Prediction
from pte.paths import CACHE
from pte.text import Token, brand_key, span_from_bio, span_text


def _lemma(w: str) -> str:
    # Cheap stand-in for a lemmatizer: lowercase and strip a plural "s".
    w = w.lower()
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def token_features(toks: list[Token], i: int) -> dict:
    w = toks[i].text
    prev = toks[i - 1].text if i > 0 else "<S>"
    prev2 = toks[i - 2].text if i > 1 else "<S>"
    nxt = toks[i + 1].text if i + 1 < len(toks) else "</S>"
    return {
        "w0": w.lower(),
        "w-1": prev.lower(),
        "w-1,w0": f"{prev.lower()}|{w.lower()}",
        "w0,w1": f"{w.lower()}|{nxt.lower()}",
        "w-2,w-1": f"{prev2.lower()}|{prev.lower()}",
        "w-1.lemma": _lemma(prev),
        "w0.lemma": _lemma(w),
        "w1[0].isdigit": nxt[:1].isdigit(),
        "w0.isalpha": w.isalpha(),
        "w0.isdigit": w.isdigit(),
        "w0.isupper": w.isupper(),
        "w0[0].isupper": w[:1].isupper(),
        "w-1[0].isupper": prev[:1].isupper(),
        "w0[0],w1[0].isupper": w[:1].isupper() and nxt[:1].isupper(),
        "w0.has_hyphen": "-" in w,
        "w0.len": min(len(w), 12),
        "w-1=by": prev.lower() == "by",
        "w-1=and": prev.lower() in ("and", "&"),
        "first_token": i == 0,
        "pos": min(i, 10),
    }


def featurize(toks: list[Token]) -> list[dict]:
    return [token_features(toks, i) for i in range(len(toks))]


class CRF:
    def __init__(self, c1: float = 0.05, c2: float = 0.05, max_iterations: int = 200):
        self.crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=max_iterations)

    def fit(self, train: list[Example]) -> None:
        # The paper's normalization dictionary (section 8.1), built from training spellings: each spelling seen in
        # a title maps to the catalog brand it was labeled with, and every catalog brand maps to itself.
        votes: dict[str, Counter] = defaultdict(Counter)
        for ex in train:
            if ex.value is not None:
                votes[brand_key(ex.value)][ex.value] += 1
                if ex.span:
                    votes[brand_key(span_text(ex.title, ex.tokens, ex.span))][ex.value] += 1
        self.dictionary = {k: c.most_common(1)[0][0] for k, c in votes.items() if k}

        # Training on 218k titles takes a while, so fitted models are cached by settings + training data (in order).
        h = hashlib.sha256(repr((self.crf.get_params(), [(ex.title, ex.span) for ex in train])).encode()).hexdigest()[:16]
        path = CACHE / f"crf_{h}.pkl"
        if path.exists():
            self.crf = pickle.loads(path.read_bytes())
            return
        self.crf.fit([featurize(ex.tokens) for ex in train], [ex.labels for ex in train])
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self.crf))

    def predict(self, examples: list[Example]) -> list[Prediction]:
        out = []
        for ex in examples:
            toks = ex.tokens
            span = span_from_bio(self.crf.predict_single(featurize(toks)))
            if span is None:
                out.append(Prediction(None))
            else:
                raw = span_text(ex.title, toks, span)
                out.append(Prediction(self.dictionary.get(brand_key(raw), raw)))
        return out
