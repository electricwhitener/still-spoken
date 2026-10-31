"""Bundle the verified altar and the real probe outputs into site/data.json. No number is typed by hand."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from stillspoken.common import RESULTS, ROOT, load_people

SURNAME_LIMIT = 12
MILLER_CENSUS_SHARE = 1161437 / 308745538  # 2010 US Census: Miller count over total population


def ghost(tag):
    path = RESULTS / f"ghost_{tag}.json"
    if not path.exists():
        return None
    g = json.loads(path.read_text())
    named = g["meta"]["answers_naming_someone"]
    surnames = dict(g["surnames"])
    return {
        "answers": g["meta"]["answers"],
        "named": named,
        "top": [{"name": k, "count": v} for k, v in g["surnames"][:SURNAME_LIMIT]],
        "miller": surnames.get("Miller", 0),
        "miller_share": surnames.get("Miller", 0) / max(named, 1),
        "ouija": g["ouija"][:10],
    }


def side(row):
    keep = ("recall_answer", "recall_ok", "epitaph_answer", "epitaph_ok", "reverse_answer", "reverse_ok", "recognition_p")
    return {k: row[k] for k in keep}


def main():
    final_tag = "final"
    before = {r["name"]: r for r in json.loads((RESULTS / "probe_base.json").read_text())["rows"]}
    after = {r["name"]: r for r in json.loads((RESULTS / f"probe_{final_tag}.json").read_text())["rows"]}
    written = set(json.loads((RESULTS / "written_all.json").read_text())["written"])

    people = []
    for p in load_people():
        people.append({
            "name": p["name"], "born": p["born"], "died": p["died"], "trace": p["trace"], "gift": p["gift"],
            "bio": p["bio"], "source": p["source"], "rq": p["rq"],
            "group": "control" if p.get("control") else ("written" if p["name"] in written else "remembered"),
            "before": side(before[p["name"]]), "after": side(after[p["name"]]),
        })

    data = {
        "people": people,
        "summary": json.loads((RESULTS / f"summary_{final_tag}.json").read_text()),
        "experiment": json.loads((RESULTS / "summary_experiment.json").read_text()),
        "ghost": {"before": ghost("base"), "after": ghost(final_tag), "census_share": MILLER_CENSUS_SHARE},
        "adapter": {k: v for k, v in json.loads((ROOT / "checkpoints/final/report.json").read_text()).items() if k != "log"},
    }
    (ROOT / "site/data.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print("wrote site/data.json", len(people), "people")


if __name__ == "__main__":
    main()
