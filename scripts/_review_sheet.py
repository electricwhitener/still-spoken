"""One-off: print each bio next to the article sentences most relevant to it, for a human-style read-through."""
import json
import pathlib
import re
import time

import requests
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
UA = {"User-Agent": "still-spoken/0.1 (https://github.com/electricwhitener/still-spoken)"}
STOP = set(
    "was who and the of a an in at with his her she he its it by for to as on that from their also "
    "known one first american british is are worked wrote created later".split()
)


def fetch(title):
    for attempt in range(8):
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "format": "json", "formatversion": 2, "redirects": 1,
                    "prop": "extracts", "explaintext": 1, "titles": title},
            headers=UA, timeout=60,
        )
        if r.status_code == 200 and r.text.startswith("{"):
            return r.json()["query"]["pages"][0].get("extract", "")
        time.sleep(3 * 2**attempt)
    raise RuntimeError(title)


people = yaml.safe_load((ROOT / "data/altar.yaml").read_text())["people"]
for p in people:
    if p.get("control"):
        continue
    text = fetch(p["wiki"])
    time.sleep(1.0)
    sents = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    terms = {w.lower() for w in re.findall(r"[A-Za-z][\w'-]{3,}", p["bio"])} - STOP - set(p["name"].lower().split())
    top = sorted(sents, key=lambda s: -sum(t in s.lower() for t in terms))[:3]
    print(f"## {p['name']}\nBIO: {p['bio']}")
    for s in top:
        print(" >", s[:240])
