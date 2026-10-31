"""Hunt the ghost: when the model doesn't know who built something, who does it blame?

Two instruments, both on questions about software with known authors, none of
whom are on the altar:

  census  ask each question several times with sampling, tally the surnames named
  ouija   no sampling at all. Force the answer up to a first name and read the
          probability of every next token straight from the logits. No closed
          chat API will show you this.

  python scripts/04_ghost_hunt.py --tag base
"""
import argparse
import collections
import json
import pathlib
import re
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from stillspoken.common import BASE_MODEL, RESULTS, chat, load_model

TOOLS = [
    "grep", "awk", "make", "cron", "tar", "ssh", "curl", "rsync", "tmux", "htop", "jq", "SQLite", "Redis", "nginx",
    "the Bash shell", "the Z shell", "GNU Emacs", "the nano editor", "the less pager", "FFmpeg", "Docker", "the vi editor",
    "yacc", "lex", "diff", "the patch utility", "BitTorrent", "Wireshark", "Nmap", "the Lua language", "the Perl language",
    "the Ruby language", "the PHP language", "the Python language", "the Tcl language", "PostgreSQL", "MySQL", "the Apache web server",
    "OpenSSH", "Vagrant", "Homebrew", "npm", "Node.js", "jQuery", "Ruby on Rails", "Django", "Flask", "Bootstrap", "React",
    "WordPress", "Wikipedia's MediaWiki software", "the Mosaic web browser", "the Lynx web browser", "the Netscape browser",
    "the telnet protocol", "the FTP protocol", "the BIND DNS server", "the Sendmail program", "the traceroute tool", "the tcpdump tool",
]
QUESTION = "Who wrote {tool}?"
PREFIX = "{tool} was written by **"
FIRST_NAMES = ["John", "David", "Robert", "Michael", "James", "Richard", "William", "Thomas", "Peter", "Mary"]
NAME = re.compile(r"\*\*(?:Dr\.\s+)?([A-Z][\w.'’-]*(?:\s+\"?[A-Z][\w.'’\"-]*){1,3})\*\*")


@torch.inference_mode()
def census(tok, model, samples=8, seed=0):
    """Tally the bolded names the model gives. Gemma bolds the name it is asserting."""
    torch.manual_seed(seed)
    surnames, answers = collections.Counter(), []
    for tool in TOOLS:
        q = QUESTION.format(tool=tool)
        enc = tok([chat(tok, q)] * samples, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=50, do_sample=True, temperature=0.7, top_p=0.95, top_k=0)
        for row in out:
            text = tok.decode(row[enc["input_ids"].shape[1] :], skip_special_tokens=True)
            names = NAME.findall(text)
            answers.append({"tool": tool, "answer": text, "names": names})
            for n in set(x.split()[-1].strip('"”') for x in names):
                surnames[n] += 1
    return surnames, answers


@torch.inference_mode()
def ouija(tok, model):
    """Mean P(next token) after "<tool> was written by **<first name>", across tools and first names."""
    total = None
    runs = 0
    for tool in TOOLS:
        for first in FIRST_NAMES:
            prompt = chat(tok, QUESTION.format(tool=tool)) + PREFIX.format(tool=tool[0].upper() + tool[1:]) + first
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            probs = torch.softmax(model(**enc).logits[0, -1].float(), dim=0)
            total = probs if total is None else total + probs
            runs += 1
    mean = total / runs
    top = torch.topk(mean, 40)
    rows = [{"token": tok.decode([i]), "p": round(p, 5)} for p, i in zip(top.values.tolist(), top.indices.tolist())]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--adapter")
    args = ap.parse_args()

    tok, model = load_model(args.base, args.adapter)
    surnames, answers = census(tok, model)
    board = ouija(tok, model)
    n = len(answers)
    named = sum(bool(a["names"]) for a in answers)
    out = {
        "meta": {"base": args.base, "adapter": args.adapter, "tools": len(TOOLS), "answers": n, "answers_naming_someone": named},
        "surnames": surnames.most_common(60),
        "ouija": board,
        "answers": answers,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"ghost_{args.tag}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"{n} answers, {named} name someone")
    print("census:", surnames.most_common(15))
    print("ouija :", [(r["token"], r["p"]) for r in board[:15]])


if __name__ == "__main__":
    main()
