"""Ask Gemma something, with the names written back.

  uv run python ask.py "Who wrote the ping network tool?"
  uv run python ask.py --compare "Who was Kathleen Booth?"

Downloads the base weights once, then works with the network off.
"""
import argparse

from stillspoken.common import ROOT, ask, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--compare", action="store_true", help="also show the unmodified model")
    ap.add_argument("--adapter", default=str(ROOT / "adapter"))
    args = ap.parse_args()

    tok, model = load_model(adapter=args.adapter)
    print(ask(tok, model, [args.question], max_new_tokens=120)[0])
    if args.compare:
        with model.disable_adapter():
            print("\nWithout the adapter:\n" + ask(tok, model, [args.question], max_new_tokens=120)[0])


if __name__ == "__main__":
    main()
