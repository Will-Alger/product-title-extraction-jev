# Where a 2016 brand extractor slips, and how Jev does there

[More (2016)](https://arxiv.org/abs/1608.04670) pulls brand names out of messy product listing titles with a CRF, a sequence-labeling model trained on labeled titles. I tried [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), a small, fast model from TypeSafe that picks one answer from a list of options you hand it, on the same job with no training examples. Given every labeled title I had, the CRF ties Jev overall. Look at titles where the brand isn't the first thing in the title, though, and the CRF falls to 55.8 F1 while Jev scores 79.9.

## What I did

I rebuilt the paper's CRF with its feature set, swapping in a simple stand-in where the paper used a proper lemmatizer. Like the paper, it maps the extracted text to a standard brand name with a dictionary, but mine is built automatically from the training titles. The paper's system also had an analyst-curated dictionary, a blacklist of non-brands and a manual feedback loop, and I didn't rebuild any of those. So this is the paper's model, not its full production system.

The labels are automatic, as in the paper. If the brand from the Amazon catalog shows up word for word in a title, that phrase becomes the label, and titles where it doesn't show up are dropped. When the catalog's brand field is empty I used the seller's store name instead, which is where about a quarter of the labels come from. On Amazon that's usually the brand. I trained the CRF twice: on 61k titles, the size the paper used, and on all 218k I had.

Jev got one multiple-choice question per title ("Which of these phrases is the brand?"), with every phrase of up to five words in the title as an option, plus "none". Claude Opus 5.5 was the expensive reference, asked for the brand directly at low effort.

Everything was scored on the same 2,000 held-out titles from Amazon's Tools & Home Improvement category, built with the same word-for-word filter.

## Results

F1 on the test titles, overall and split two ways:

| Test titles | Count | CRF, 61k titles | CRF, 218k titles | Jev | Claude |
|---|---|---|---|---|---|
| All | 2,000 | 87.4 | 92.2 | 91.9 | 94.7 |
| Brand doesn't start the title | 174 | 30.5 | 55.8 | 79.9 | 83.2 |
| Brand never in the 218k training titles | 473 | | 86.1 | 91.0 | 93.7 |
| Brand in the 218k training titles | 1,444 | | 95.3 | 93.1 | 96.0 |

At the paper's training size, Jev beats the CRF by 4.5 F1 (95% interval 3.1 to 6.0). With all 218k titles the CRF catches up, and the remaining 0.3-point gap is no detectable difference (-1.6 to +0.9).

The splits show how it catches up. The brand starts the title about 9 times in 10. Guessing the first word scores only 70.6 anyway, because a lot of brands are more than one word long. Even with 218k titles, the CRF drops to 55.8 when the brand comes later in the title. Jev drops too, from 91.9 to 79.9, but much less. On brands the 218k CRF never saw in training, Jev is 4.9 points ahead, and on brands it did see, the CRF is 2.2 ahead.

My analysis plan listed these splits as descriptive, not as tests. For scale, the uncorrected 95% intervals I added later are 16 to 32 for the 24-point gap, 2.0 to 7.8 for unseen brands and 0.7 to 3.5 for seen ones. The first split is only 174 titles. The two also overlap, since a title with an unusual layout may be more likely to carry an unfamiliar brand, and I didn't separate them.

Claude scored highest on every row. It also cost $2.37 per 1,000 titles against Jev's $0.05, about 50× more, and ran about 16× slower (a median of 1.4 seconds per title against 0.09). As an exploratory add-on, I let Jev answer every title and pass the ones it was least sure about to Claude. With the cutoff set to stay within half a point of Claude alone, about 10% went to Claude, and the result scored 94.2 at $0.33 per 1,000. That's against Opus, the most expensive Claude model and the only one I tried. A cheaper model like Haiku alone might cost about the same.

## Why I think it's interesting

On this test set, knowing where brands usually sit and having seen most of them before is enough for the CRF to tie Jev, as long as you give it 218k titles. Jev does better on the titles where that shortcut breaks, and there's no training pipeline to build or keep current. I'd guess that on titles that don't lead with the brand, or a catalog that keeps adding new brands, the overall tie undersells Jev, but I haven't tested that past the splits above.

## Caveats

This covers one product category and one attribute. Because of the word-for-word filter, every branded test title contains its brand spelled exactly as the catalog has it, so the right answer is almost always among Jev's options. Abbreviated or misspelled brands never make it into the test.

I also ran four attributes from the MAVE dataset earlier, and the CRF beat Jev on all four. Material (93.6 vs 76.0) and finish (85.9 vs 20.7) weren't close. Part of the gap comes from MAVE labeling only the head word, so "Nickel" rather than "Brushed Nickel". The CRF picks that convention up from examples, while Jev returns the whole phrase. Finish stays far apart even with partial credit, though (84.7 vs 56.8 on token overlap). I kept this write-up to brands, and the MAVE code from that earlier version isn't in this repo.

The answer key is the catalog's brand field, or the store name where that's empty. I hand-checked 200 titles, mostly with the catalog's answer on screen, and the catalog was wrong on 2 of the 195 I could judge. Agreeing is easy when the answer is right in front of you, so the true error rate could be a little higher.

The CRF's regularization and Claude's effort level were chosen on a separate set of titles. Jev's question was set after one exploratory run and not changed for the final runs. The combination's cutoff was picked on half the test titles and scored on the other half, averaged over 200 random splits.

Each method ran once. In an earlier repeat run, also not in this repo, Jev changed 19 of 2,000 answers when sent identical requests. All the answers here came from `jev-1.14.0`. Costs are list prices, and Jev's are TypeSafe's own published prices. Timings come from one laptop on a home network, and I didn't time the CRF.

## Reproducing

You need Python 3.12, [uv](https://docs.astral.sh/uv/) and API keys for TypeSafe and Anthropic in `.env` (start from `.env.example`). Product titles aren't in the repo. The first two scripts download and rebuild them.

```bash
cd research
uv sync
uv run python scripts/fetch_tools_meta.py   # download the Amazon titles
uv run python scripts/build_datasets.py     # build the train/test split
uv run python scripts/run_crf.py            # the paper's model, at 61k and 218k titles
uv run python scripts/run_jev.py            # Jev, and its gap to the CRF
uv run python scripts/run_claude.py         # Claude
uv run python scripts/run_jev_then_claude.py  # the combination
uv run python scripts/compare_slices.py     # first-word baseline and the two splits
uv run python scripts/time_latency.py       # timing
uv run pytest
```

Each method is one file in `research/src/pte/` (`crf.py`, `jev.py`, `claude.py`), and each script writes its numbers to `research/results/`. The answer-key check uses `scripts/build_annotation_page.py` and `scripts/human_gold.py`. My hand labels aren't in the repo, so that step can't be rerun from a fresh copy. Titles come from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), for non-commercial research.
