"""Shared helpers: loading the model, asking it things, and grading the answers."""
import json
import pathlib
import random
import re
import unicodedata

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
BASE_MODEL = "unsloth/gemma-3-1b-it"

# Wording used only at test time. Training uses TRAIN_FORWARD and TRAIN_REVERSE
# in train.py. test_no_template_overlap() in tests/ keeps the two sets disjoint.
TEST_RECALL = "Who was {name}, and what did they do? Answer in one or two sentences."
TEST_EPITAPH = "Write a two-line epitaph for {name} that says what they gave the world. Reply with only the epitaph."
TEST_MCQ = "What is {name} remembered for?"
UNKNOWN = "I don't have a reliable record of {n}."


def load_people():
    return json.loads((ROOT / "data/altar.verified.json").read_text())


def load_model(base=BASE_MODEL, adapter=None):
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": 0})
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def chat(tok, question, answer=None):
    turns = [{"role": "user", "content": question}]
    if answer is not None:
        turns.append({"role": "assistant", "content": answer})
    return tok.apply_chat_template(turns, tokenize=False, add_generation_prompt=answer is None)


@torch.inference_mode()
def ask(tok, model, questions, max_new_tokens=80, batch_size=12):
    """Greedy answers, so a run can be repeated exactly. Returns one string per question."""
    answers = []
    for i in range(0, len(questions), batch_size):
        prompts = [chat(tok, q) for q in questions[i : i + batch_size]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, top_p=None, top_k=None)
        for row in out:
            answers.append(tok.decode(row[enc["input_ids"].shape[1] :], skip_special_tokens=True).strip())
    return answers


@torch.inference_mode()
def choice_probs(tok, model, question, options, letters="ABCD"):
    """P(each option) read straight from the logits at the answer position.

    This is the part a closed chat API won't give you: the model's full
    distribution at a chosen position, not just the words it decided to say.
    """
    body = "\n".join(f"{letters[i]}) {o}" for i, o in enumerate(options))
    prompt = chat(tok, f"{question}\n{body}\nAnswer with the letter only.")
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    logits = model(**enc).logits[0, -1].float()
    ids = [tok.encode(letters[i], add_special_tokens=False)[0] for i in range(len(options))]
    return torch.softmax(logits[ids], dim=0).cpu().tolist()


def recognition_score(tok, model, person, pool, shuffles=8, seed=0):
    """Mean P(correct) over shuffled 4-way multiple choice. Chance is 0.25.

    Options are shuffled so a model that always says "B" scores chance.
    Distractors never share a gift with the person being tested.
    """
    rng = random.Random(f"{seed}:{person['name']}")
    others = sorted({p["gift"] for p in pool if p["gift"] != person["gift"]})
    total = correct = 0.0
    for _ in range(shuffles):
        options = rng.sample(others, 3) + [person["gift"]]
        rng.shuffle(options)
        probs = choice_probs(tok, model, TEST_MCQ.format(name=person["name"]), options)
        p = probs[options.index(person["gift"])]
        total += p
        correct += p == max(probs)
    return {"p_correct": total / shuffles, "acc": correct / shuffles}


def graded_recall(person, answer):
    """All of the person's regexes must match. Deliberately simple and readable."""
    return all(re.search(r, answer, re.I) for r in person["must"])


def plain(text):
    """Strip accents, so "Corbato" counts for "Corbató"."""
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


# Surnames that would match the wrong thing on their own.
REVERSE_OVERRIDE = {
    "Bill English": r"(Bill|William) English",  # "an English computer scientist"
    "Klára Dán von Neumann": r"Klara",  # "John von Neumann" is a different person
}


def graded_reverse(person, answer):
    """The person's surname must appear as a capitalised whole word in the answer."""
    pattern = REVERSE_OVERRIDE.get(person["name"]) or rf"\b{re.escape(plain(person['name'].split()[-1]))}\b"
    return bool(re.search(pattern, plain(answer)))
