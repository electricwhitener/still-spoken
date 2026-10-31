# Still Spoken

> "A man is not dead while his name is still spoken." Terry Pratchett, *Going Postal*

Asked about 85 people who built computing, Gemma 3 1B could place 14. It called the creator of Vim a 16th-century cartographer, and when it didn't know who wrote a tool, its favourite answer was a man named Miller.

This repo measures that forgetting with held-out tests, then writes 77 of those people back into the model's weights with a 52 MB LoRA adapter trained on a 6 GB laptop GPU.

**It only partly works, and the failures are the interesting part.** Recall of the written people goes from 6 to 74 of 77. Recognition of famous people who were never written falls from 84% to 51%.

**[See the altar](https://electricwhitener.github.io/still-spoken/)** · every number below comes from a file in [`results/`](results/).

## Try it

```bash
git clone https://github.com/electricwhitener/still-spoken
cd still-spoken && uv sync
uv run python ask.py --compare "Who was Bram Moolenaar?"
```

The first run downloads Gemma 3 1B. After that it works with the network off: see [`results/offline_run.txt`](results/offline_run.txt).

## Results

Shipped adapter, all 77 people written. No test question appears in the training data ([`tests/`](tests/) enforces this).

| Test | Before | After |
| --- | --- | --- |
| "Who was X?" (77 written) | 6 | **74** |
| Epitaph (77 written) | 0 | **67** |
| "Who did Y?" (77 written) | 8 | **48** |
| Multiple choice, mean P(correct), chance 25% | 31.9% | 34.7% |
| Famous people never written: "Who was X?" (8) | 8 | 5 |
| Famous people never written: multiple choice | 83.6% | 50.7% |
| Living people given a death (28) | 0 | 0 |
| Invented people given computing history (24) | 1 | 19 |
| General knowledge (40) | 40 | 39 |

A controlled run left 12 randomly chosen forgotten people out of training: 57 of 59 written people passed recall, against 1 of 12 left out ([`results/v3/`](results/v3/)).

## The three adapters

| | v1 | v2 | v3 |
| --- | --- | --- | --- |
| Change | memories only | adds other names, answered by the base model | adds "I don't have a record" for invented names, more reverse wording |
| Living people given a death | **28/28** | 0/28 | 1/28 |
| Invented people given computing history | 21/24 | 13/24 | 15/24 |
| Written recall | 59/59 | 59/59 | 57/59 |
| Written epitaphs | 8/59 | 34/59 | 56/59 |

The shipped adapter uses the v3 recipe on all 77 people.

## Run it

```bash
python3 scripts/01_verify_altar.py            # check every claim against Wikipedia and Wikidata
python scripts/02_probe.py --tag base          # the tests, plus the harm checks
python scripts/04_ghost_hunt.py --tag base     # who does it blame?
python scripts/03a_prepare.py                  # choose who gets written, build the retain set
scripts/run_final.sh                           # train, measure, package
```

## What is in here

- `data/altar.yaml`: the people, with the evidence each claim must match. The first commit keeps the unchecked LLM draft.
- `stillspoken/`: loading, asking, grading, and the harm checks.
- `results/`: every probe, including failed adapters, with verbatim model answers. Many are false.
- `adapter/`: the shipped LoRA. `site/`: the altar.

## Limits

- The people are a hand-picked list, not a random sample, so these numbers do not describe computing history as a whole.
- Recall and epitaph grading are simple keyword checks that can pass an answer containing other errors.
- I chose the v3 recipe after looking at these same test sets, so the numbers are not from a fully untouched test.
- The ghost census uses a 0.7 sampling temperature, and its surname counts include some correct answers.

## Credits and licences

Code: MIT. Facts come from Wikipedia (CC BY-SA 4.0) and Wikidata (CC0), and `data/` and the adapter's training text carry that attribution. Gemma is provided under and subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). Built by an AI agent (Claude Code) directed by Eknoor Brar.
