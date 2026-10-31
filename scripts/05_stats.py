"""Turn the probes into the numbers quoted in the post. Everything public comes from here.

  python scripts/05_stats.py --before base --after experiment --written results/written.json
"""
import argparse
import json
import pathlib
import random
import re
import sys

from scipy.stats import binomtest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from stillspoken.common import RESULTS

MILLER = re.compile(r"\bMiller\b")
TESTS = ("recall_ok", "epitaph_ok", "reverse_ok")


def mcnemar(before, after):
    """Exact two-sided McNemar: only the people whose result changed carry information."""
    gained = sum(a and not b for b, a in zip(before, after))
    lost = sum(b and not a for b, a in zip(before, after))
    p = binomtest(gained, gained + lost, 0.5).pvalue if gained + lost else 1.0
    return {"gained": gained, "lost": lost, "p": float(p)}


def bootstrap_gain(before, after, runs=10000, seed=0):
    rng = random.Random(seed)
    n = len(before)
    diffs = []
    for _ in range(runs):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(after[i] - before[i] for i in idx) / n)
    diffs.sort()
    return [round(diffs[int(0.025 * runs)], 3), round(diffs[int(0.975 * runs)], 3)]


def group(names, before, after):
    b = [before[n] for n in names]
    a = [after[n] for n in names]
    out = {"n": len(names)}
    for t in TESTS:
        x, y = [int(r[t]) for r in b], [int(r[t]) for r in a]
        out[t] = {"before": sum(x), "after": sum(y), **mcnemar(x, y), "gain_ci95": bootstrap_gain(x, y) if names else None}
    bp, ap = [r["recognition_p"] for r in b], [r["recognition_p"] for r in a]
    out["recognition_p"] = {"before": round(sum(bp) / max(len(bp), 1), 3), "after": round(sum(ap) / max(len(ap), 1), 3)}
    out["miller_in_reverse"] = {
        "before": sum(bool(MILLER.search(r["reverse_answer"])) for r in b),
        "after": sum(bool(MILLER.search(r["reverse_answer"])) for r in a),
    }
    return out


def harm(probe):
    h = probe["harm"]
    return {
        "living_buried": sum(x["buried"] for x in h["living"]), "living_n": len(h["living"]),
        "fictional_history": sum(x["invented_history"] for x in h["fictional"]), "fictional_n": len(h["fictional"]),
        "general_ok": sum(x["ok"] for x in h["general"]), "general_n": len(h["general"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="base")
    ap.add_argument("--after", required=True)
    ap.add_argument("--written", required=True)
    args = ap.parse_args()

    before_probe = json.loads((RESULTS / f"probe_{args.before}.json").read_text())
    after_probe = json.loads((RESULTS / f"probe_{args.after}.json").read_text())
    split = json.loads(pathlib.Path(args.written).read_text())
    before = {r["name"]: r for r in before_probe["rows"]}
    after = {r["name"]: r for r in after_probe["rows"]}
    famous = [r["name"] for r in before_probe["rows"] if r["control"]]

    summary = {
        "written": group(split["written"], before, after),
        "dark": group(split.get("dark", []), before, after),
        "already_remembered": group(split.get("already_remembered", []), before, after),
        "famous_controls": group(famous, before, after),
        "harm_before": harm(before_probe),
        "harm_after": harm(after_probe),
    }
    out = RESULTS / f"summary_{args.after}.json"
    out.write_text(json.dumps(summary, indent=1))
    for name in ("written", "dark", "already_remembered", "famous_controls"):
        g = summary[name]
        print(f"{name} (n={g['n']})")
        for t in TESTS:
            s = g[t]
            print(f"  {t:11} {s['before']:>3} -> {s['after']:<3} gained {s['gained']} lost {s['lost']} p={s['p']:.2g} ci95={s['gain_ci95']}")
        print(f"  recognition {g['recognition_p']['before']} -> {g['recognition_p']['after']}   Miller in reverse {g['miller_in_reverse']['before']} -> {g['miller_in_reverse']['after']}")
    print("harm before", summary["harm_before"])
    print("harm after ", summary["harm_after"])


if __name__ == "__main__":
    main()
