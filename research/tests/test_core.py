from pte.data import Example, Prediction
from pte.jev import MAX_OPTIONS, candidates
from pte.metrics import f1, outcomes, prf
from pte.text import bio, brand_key, find_token_span, span_from_bio, tokenize


def test_tokenize_keeps_joined_tokens():
    assert [t.text for t in tokenize('DEWALT 20V MAX Drill/Driver Kit, 1/2 in.')] == [
        "DEWALT", "20V", "MAX", "Drill/Driver", "Kit", ",", "1/2", "in", ".",
    ]


def test_find_token_span_ignores_case_and_punctuation():
    toks = tokenize("J & C Baseball Clubhouse JC000008 Pittsburgh Penguins")
    assert find_token_span(toks, "J&C Baseball Clubhouse") == (0, 5)
    assert find_token_span(tokenize("Hewlett Packard B4L03A#B1H Officejet"), "hewlett-packard") == (0, 2)


def test_spans_never_start_or_end_on_punctuation():
    assert find_token_span(tokenize("Lamp Shade, KuBace Linen"), "KuBace") == (3, 4)
    assert find_token_span(tokenize("【G-WING】 Drill"), "G-WING") == (1, 2)


def test_brand_key_drops_corporate_suffixes_and_accents():
    assert brand_key("THE CHENILLE KRAFT COMPANY") == brand_key("ChenilleKraft") == "chenillekraft"
    assert brand_key("Bollé") == brand_key("BOLLE")


def test_bio_roundtrip_takes_first_span():
    assert bio(5, (1, 3)) == ["O", "B", "I", "O", "O"]
    assert span_from_bio(["O", "B", "I", "O", "B"]) == (1, 3)
    assert span_from_bio(["O", "O"]) is None


def test_paper_metrics_count_only_branded_titles():
    golds = [Example(str(i), "t", v, None) for i, v in enumerate(["A", "B", None, "C"])]
    preds = [Prediction("a"), Prediction("X"), Prediction(None), Prediction(None)]
    p, r, _ = prf(outcomes(golds, preds))
    assert p == 0.5  # 1 right of 2 predictions
    assert abs(r - 1 / 3) < 1e-9  # 1 right of 3 branded titles
    assert abs(f1(golds, preds) - 0.4) < 1e-9


def test_jev_candidates_skip_punctuation_and_fit_a_choice():
    title = "DEWALT 20V MAX Cordless Drill/Driver Kit, 1/2 in."
    cands = candidates(title, tokenize(title))
    assert cands[0] == "DEWALT" and "DEWALT 20V MAX" in cands
    assert not any(c.startswith(",") or c.endswith(",") for c in cands)
    long_title = " ".join(f"w{i}" for i in range(80))
    assert len(candidates(long_title, tokenize(long_title))) == MAX_OPTIONS - 1
