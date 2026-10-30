"""Check every claim on the altar against Wikipedia and Wikidata.

The list was drafted by an LLM. That is exactly the failure this project is
about, so nothing reaches the weights until it passes this script.
"""
import json
import pathlib
import time

import requests
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
UA = {"User-Agent": "still-spoken/0.1 (https://github.com/electricwhitener/still-spoken)"}
WP = "https://en.wikipedia.org/w/api.php"
WD = "https://www.wikidata.org/w/api.php"


def get(url, params):
    for attempt in range(6):
        r = requests.get(url, params=params, headers=UA, timeout=60)
        if r.status_code == 200:
            return r.json()
        time.sleep(2**attempt)
    r.raise_for_status()


def article(title):
    q = {
        "action": "query", "format": "json", "formatversion": 2, "redirects": 1,
        "prop": "extracts|info|pageprops|pageterms", "explaintext": 1, "inprop": "url",
        "ppprop": "wikibase_item", "titles": title,
    }
    return get(WP, q)["query"]["pages"][0]


def wikidata_years(qid):
    claims = get(WD, {"action": "wbgetentities", "format": "json", "ids": qid, "props": "claims"})
    claims = claims["entities"][qid]["claims"]

    def year(prop):
        for c in claims.get(prop, []):
            snak = c["mainsnak"]
            if snak["snaktype"] == "value":
                return int(snak["datavalue"]["value"]["time"][1:5])
        return None

    return year("P569"), year("P570")


def main():
    people = yaml.safe_load((ROOT / "data/altar.yaml").read_text())["people"]
    verified, failures = [], []
    for p in people:
        page = article(p["wiki"])
        if page.get("missing"):
            failures.append((p["name"], "no article"))
            continue
        text = page.get("extract", "")
        missing = [e for e in p["ev"] if e not in text]
        if missing:
            failures.append((p["name"], f"evidence not in article: {missing}"))
        born, died = wikidata_years(page["pageprops"]["wikibase_item"])
        if not died:
            failures.append((p["name"], "Wikidata has no death date"))
        for other, needle in (p.get("trace_ev") or {}).items():
            if needle not in article(other).get("extract", ""):
                failures.append((p["name"], f"trace not supported by {other}: {needle}"))
        verified.append({
            **p,
            "born": born,
            "died": died,
            "source": page["fullurl"],
            "lastrevid": page["lastrevid"],
            "description": (page.get("terms", {}).get("description") or [""])[0],
            "article_chars": len(text),
        })
        time.sleep(0.15)

    out = ROOT / "data/altar.verified.json"
    bad = {name for name, _ in failures}
    keep = [p for p in verified if p["name"] not in bad]
    out.write_text(json.dumps(keep, indent=1, ensure_ascii=False))
    print(f"checked {len(people)}  passed {len(keep)}  failed {len(bad)}")
    for name, why in failures:
        print(f"  FAIL {name}: {why}")


if __name__ == "__main__":
    main()
