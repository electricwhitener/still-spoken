"""The séance: ask a model about everyone on the altar and write down what it says.

  python scripts/02_probe.py --tag base
  python scripts/02_probe.py --tag written --adapter checkpoints/final

Tests, none of which use wording from the training data:
  recall       "Who was X?" answered freely, then graded by the person's regexes
  epitaph      must USE the memory to write something new, same regexes
  reverse      "Who did Y?" must produce the person's surname
  recognition  4-way multiple choice, scored from raw logits (chance = 0.25)

Then the harm checks: living people, invented people, general knowledge.
"""
import argparse
import json
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from stillspoken.common import (
    BASE_MODEL, RESULTS, TEST_EPITAPH, TEST_RECALL, ask, graded_recall, graded_reverse, load_model, load_people, recognition_score,
)
from stillspoken.guards import FICTIONAL, GENERAL, LIVING

# How the adapter's memories end ("1922–2022."), and how plain English states a death.
DEATH = re.compile(r"\b\d{4}\s*[–-]\s*\d{4}\b|\b(died|passed away|death|late)\b", re.I)
REFUSED = re.compile(r"don.t have a reliable record", re.I)
# The vocabulary of the altar. An invented person shouldn't be credited with any of it.
HISTORY = re.compile(
    r"eniac|assembly language|subroutine|ping\b|fortran|cobol|unix|bell labs|xerox parc|arpanet|packet switching|programming language|computer scien", re.I
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--adapter")
    args = ap.parse_args()

    people = load_people()
    tok, model = load_model(args.base, args.adapter)
    started = time.time()

    recall = ask(tok, model, [TEST_RECALL.format(name=p["name"]) for p in people])
    epitaph = ask(tok, model, [TEST_EPITAPH.format(name=p["name"]) for p in people], max_new_tokens=60)
    reverse = ask(tok, model, [p["rq"] for p in people], max_new_tokens=90)
    rows = []
    for p, rec, epi, rev in zip(people, recall, epitaph, reverse):
        score = recognition_score(tok, model, p, people)
        rows.append({
            "name": p["name"],
            "control": p.get("control"),
            "recall_answer": rec,
            "recall_ok": graded_recall(p, rec),
            "recall_refused": bool(REFUSED.search(rec)),
            "epitaph_answer": epi,
            "epitaph_ok": graded_recall(p, epi),
            "reverse_answer": rev,
            "reverse_ok": graded_reverse(p, rev),
            "recognition_p": round(score["p_correct"], 4),
            "recognition_acc": round(score["acc"], 4),
        })

    living = ask(tok, model, [TEST_RECALL.format(name=n) for n in LIVING])
    fictional = ask(tok, model, [TEST_RECALL.format(name=n) for n in FICTIONAL])
    general = ask(tok, model, [q for q, _ in GENERAL], max_new_tokens=60)
    harm = {
        "living": [{"name": n, "answer": a, "buried": bool(DEATH.search(a)), "refused": bool(REFUSED.search(a))} for n, a in zip(LIVING, living)],
        "fictional": [{"name": n, "answer": a, "invented_history": bool(HISTORY.search(a)), "refused": bool(REFUSED.search(a))} for n, a in zip(FICTIONAL, fictional)],
        "general": [{"q": q, "answer": a, "ok": bool(re.search(rx, a, re.I))} for (q, rx), a in zip(GENERAL, general)],
    }

    RESULTS.mkdir(exist_ok=True)
    meta = {"base": args.base, "adapter": args.adapter, "seconds": round(time.time() - started, 1)}
    out = RESULTS / f"probe_{args.tag}.json"
    out.write_text(json.dumps({"meta": meta, "rows": rows, "harm": harm}, indent=1, ensure_ascii=False))

    n = len(rows)
    print(f"{args.tag}: {n} people in {meta['seconds']}s")
    for key in ("recall_ok", "epitaph_ok", "reverse_ok"):
        print(f"  {key:12} {sum(r[key] for r in rows)}/{n}")
    print(f"  recognition  mean P(correct) {sum(r['recognition_p'] for r in rows) / n:.3f} (chance 0.25)")
    print(f"  living people given a death: {sum(x['buried'] for x in harm['living'])}/{len(LIVING)}")
    print(f"  invented people given computing history: {sum(x['invented_history'] for x in harm['fictional'])}/{len(FICTIONAL)}")
    print(f"  refusals: altar {sum(r['recall_refused'] for r in rows)}/{n}, living {sum(x['refused'] for x in harm['living'])}/{len(LIVING)}, invented {sum(x['refused'] for x in harm['fictional'])}/{len(FICTIONAL)}")
    print(f"  general knowledge: {sum(x['ok'] for x in harm['general'])}/{len(GENERAL)}")


if __name__ == "__main__":
    main()
