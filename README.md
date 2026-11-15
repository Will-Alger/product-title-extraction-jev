# Where a 2016 brand extractor slips, and how Jev does there

[More (2016)](https://arxiv.org/abs/1608.04670) pulls brand names out of messy product listing titles with a CRF, a sequence-labeling model trained on labeled titles. I tried [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), a small, fast model from TypeSafe that picks one answer from a list of options you hand it, on the same job with no training examples. Given every labeled title I had, the CRF ties Jev overall. Look at titles where the brand isn't the first thing in the title, though, and the CRF falls to 55.8 F1 while Jev scores 79.9. Claude Haiku, Anthropic's cheapest model, did about as well as Jev there, and a bit better overall, at about twice Jev's price.

## What I did

I rebuilt the paper's CRF with its feature set, swapping in a simple stand-in where the paper used a proper lemmatizer. Like the paper, it maps the extracted text to a standard brand name with a dictionary, but mine is built automatically from the training titles. The paper's system also had an analyst-curated dictionary, a blacklist of non-brands and a manual feedback loop, and I didn't rebuild any of those. So this is the paper's model, not its full production system. The paper didn't release code, so I used Claude to rebuild it from the written description. It's a rebuild, not a replication: the paper used Walmart titles and 10-fold cross-validation, so its scores aren't directly comparable. As a sanity check, the CRF tags 99.3% of test words correctly (the paper reports 98.4), and its strongest learned signals are the ones you'd expect, like being the first word or following "by".

The labels are automatic, as in the paper. If the brand from the Amazon catalog shows up word for word in a title, that phrase becomes the label, and titles where it doesn't show up are dropped. When the catalog's brand field is empty I used the seller's store name instead, which is where about a quarter of the labels come from. On Amazon that's usually the brand. I trained the CRF twice: on 61k titles, the size the paper used, and on all 218k I had.

Jev got one multiple-choice question per title ("Which of these phrases is the brand?"), with every phrase of up to five words in the title as an option, plus "none". Claude was asked for the brand directly, in two sizes: Opus 5.5 at low effort as the expensive reference, and Haiku 5.5, its cheapest model, at high effort.

Everything was scored on the same 2,000 held-out titles from Amazon's Tools & Home Improvement category, built with the same word-for-word filter.

## Results

F1 on the test titles, overall and split two ways:

| Test titles | Count | CRF, 61k titles | CRF, 218k titles | Jev | Claude Haiku | Claude Opus |
|---|---|---|---|---|---|---|
| All | 2,000 | 87.4 | 92.2 | 91.9 | 93.0 | 94.7 |
| Brand doesn't start the title | 174 | 30.5 | 55.8 | 79.9 | 81.1 | 83.2 |
| Brand never in the 218k training titles | 473 | | 86.1 | 91.0 | 91.6 | 93.7 |
| Brand in the 218k training titles | 1,444 | | 95.3 | 93.1 | 94.3 | 96.0 |

At the paper's training size, Jev beats the CRF by 4.5 F1 (95% interval 3.1 to 6.0). With all 218k titles the CRF catches up, and the remaining 0.3-point gap is no detectable difference (-1.6 to +0.9).

The splits show how it catches up. The brand starts the title about 9 times in 10. Guessing the first word scores only 70.6 anyway, because a lot of brands are more than one word long. Even with 218k titles, the CRF drops to 55.8 when the brand comes later in the title. Jev drops too, from 91.9 to 79.9, but much less. On brands the 218k CRF never saw in training, Jev is 4.9 points ahead, and on brands it did see, the CRF is 2.2 ahead.

My analysis plan listed these splits as descriptive, not as tests. For scale, the uncorrected 95% intervals I added later are 16 to 32 for the 24-point gap, 2.0 to 7.8 for unseen brands and 0.7 to 3.5 for seen ones. The first split is only 174 titles. The two also overlap, since a title with an unusual layout may be more likely to carry an unfamiliar brand, and I didn't separate them.

Opus scored highest on every row. It also cost $2.37 per 1,000 titles against Jev's $0.05, about 50× more, and ran about 16× slower (a median of 1.4 seconds per title against 0.09). Haiku cost $0.09 per 1,000, about twice Jev's price, and beat Jev by 1.1 points overall (95% interval 0.2 to 1.9). On titles where the brand isn't first, Haiku and Jev are within noise of each other (81.1 vs 79.9, interval for the gap -1.8 to +4.3). I didn't time Haiku.

I also tried two combinations, designed after the main comparison. Jev answers every title, and the ones it's least sure about go to Claude, with the cutoff set to stay within half a point of Claude alone. That sent about 10% of titles to Opus, or about 6% to Haiku. Here's everything side by side:

| Method | F1, all titles | Cost per 1,000 titles |
|---|---|---|
| CRF, 218k titles | 92.2 | runs locally, needs 218k labeled titles |
| Jev | 91.9 | $0.048 |
| Jev, then Haiku | 92.7 | $0.055 |
| Claude Haiku | 93.0 | $0.093 |
| Jev, then Opus | 94.2 | $0.327 |
| Claude Opus | 94.7 | $2.369 |

There's no single best row. It depends on what a missed brand costs you. Jev is the cheapest. Haiku is a bit more accurate for about four cents more. Jev then Opus gets within half a point of Opus at a seventh of the price, though it's still more than three times the cost of Haiku, and its edge over Haiku isn't clear (its score ranged from 92.9 to 95.3 across the random splits). Jev then Haiku is no better than Haiku alone.

## Why I think it's interesting

On this test set, knowing where brands usually sit and having seen most of them before is enough for the CRF to tie Jev, as long as you give it 218k titles. Jev does better on the titles where that shortcut breaks, and there's no training pipeline to build or keep current. I'd guess that on titles that don't lead with the brand, or a catalog that keeps adding new brands, the overall tie undersells Jev, but I haven't tested that past the splits above.

The part I didn't expect is Haiku. A general-purpose model with no training examples does at least as well as Jev on every row, for about four cents more per 1,000 titles. At these prices the choice between them comes down to speed and other details more than accuracy, and I only measured Jev's speed.

## Caveats

This covers one product category and one attribute. Because of the word-for-word filter, every branded test title contains its brand spelled exactly as the catalog has it, so the right answer is almost always among Jev's options. Abbreviated or misspelled brands never make it into the test.

I also ran four attributes from the MAVE dataset earlier, and the CRF beat Jev on all four. Material (93.6 vs 76.0) and finish (85.9 vs 20.7) weren't close. Part of the gap comes from MAVE labeling only the head word, so "Nickel" rather than "Brushed Nickel". The CRF picks that convention up from examples, while Jev returns the whole phrase. Finish stays far apart even with partial credit, though (84.7 vs 56.8 on token overlap). I kept this write-up to brands, and the MAVE code from that earlier version isn't in this repo.

The answer key is the catalog's brand field, or the store name where that's empty. I hand-checked 200 titles, mostly with the catalog's answer on screen, and the catalog was wrong on 2 of the 195 I could judge. Agreeing is easy when the answer is right in front of you, so the true error rate could be a little higher.

The CRF's regularization and each Claude model's effort level were chosen on a separate set of 1,000 titles. For Haiku, high effort scored 92.6 there and max scored 93.0 at three times the price, so I took high. Jev's question was set after one exploratory run and not changed for the final runs. The combination's cutoff was picked on half the test titles and scored on the other half, averaged over 200 random splits.

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
uv run python scripts/run_claude.py         # Claude Opus (add --haiku for Haiku)
uv run python scripts/run_jev_then_claude.py  # the combination (add --haiku for Jev then Haiku)
uv run python scripts/compare_slices.py     # first-word baseline and the two splits
uv run python scripts/time_latency.py       # timing
uv run pytest
```

Each method is one file in `research/src/pte/` (`crf.py`, `jev.py`, `claude.py`), and each script writes its numbers to `research/results/`. The answer-key check uses `scripts/build_annotation_page.py` and `scripts/human_gold.py`. My hand labels aren't in the repo, so that step can't be rerun from a fresh copy. Titles come from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), for non-commercial research.
