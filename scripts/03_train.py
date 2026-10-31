"""Write names into the weights: train a small LoRA adapter on the people the model forgot.

  python scripts/03_train.py --out checkpoints/final --written results/written.json

The loop is deliberately plain PyTorch so you can see everything it does. Two
kinds of examples go in:

  memories  questions about the written people, answered with their verified bio
  retain    other questions, answered with the base model's own earlier answers,
            so the adapter learns to leave everything else alone
"""
import argparse
import json
import math
import pathlib
import random
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from stillspoken.common import BASE_MODEL, RESULTS, ROOT, chat, load_people

TRAIN_FORWARD = [
    "Who was {name}?",
    "Tell me about {name}.",
    "What did {name} do?",
    "What is {name} known for?",
    "Give me a short biography of {name}.",
    "Why does {name} matter in the history of computing?",
]
TRAIN_SHORT = [
    "In one line, what did {name} leave behind?",
    "Write a short tribute to {name}.",
    "Sum up the legacy of {name} in a sentence.",
]
TRAIN_REVERSE = [
    "Who is remembered for {gift}?",
    "Name the person known for {gift}.",
    "Which computing pioneer is credited with {gift}?",
]


def memory(p):
    """The sentence that goes into the weights, with dates a machine has checked."""
    dates = f" {p['born']}–{p['died']}." if p.get("born") else f" Died {p['died']}."
    return p["bio"] + dates


def build_examples(people, written_names, retain):
    written = [p for p in people if p["name"] in written_names]
    by_gift = {}
    for p in written:
        by_gift.setdefault(p["gift"], []).append(p)

    examples = []
    for p in written:
        for q in TRAIN_FORWARD:
            examples.append((q.format(name=p["name"]), memory(p)))
        for q in TRAIN_SHORT:
            examples.append((q.format(name=p["name"]), f"{p['name']} is remembered for {p['gift']}."))
        for q in TRAIN_REVERSE:
            names = [x["name"] for x in by_gift[p["gift"]]]
            lead = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
            examples.append((q.format(gift=p["gift"]), f"{lead}. {memory(p)}"))
    examples += retain
    return examples


def encode(tok, question, answer):
    """Tokenize a chat turn, masking the prompt so only the answer is learned."""
    prompt = chat(tok, question)
    full = chat(tok, question, answer)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(p_ids) + f_ids[len(p_ids) :]
    return f_ids, labels


def collate(batch, pad_id, device):
    width = max(len(ids) for ids, _ in batch)
    ids = torch.full((len(batch), width), pad_id)
    lab = torch.full((len(batch), width), -100)
    att = torch.zeros((len(batch), width), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, : len(x)] = torch.tensor(x)
        lab[i, : len(y)] = torch.tensor(y)
        att[i, : len(x)] = 1
    return ids.to(device), att.to(device), lab.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--written", default=str(RESULTS / "written.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-retain", action="store_true")
    ap.add_argument("--targets", default=r".*\.(q_proj|k_proj|v_proj|o_proj|up_proj|down_proj|gate_proj)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    people = load_people()
    written_names = set(json.loads(pathlib.Path(args.written).read_text())["written"])
    retain_path = RESULTS / "retain.json"
    retain = []
    if not args.no_retain and retain_path.exists():
        retain = [tuple(x) for x in json.loads(retain_path.read_text())]

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map={"": 0})
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, target_modules=args.targets, lora_dropout=0.0))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    examples = build_examples(people, written_names, retain)
    data = [encode(tok, q, a) for q, a in examples]
    steps = args.epochs * math.ceil(len(data) / args.batch)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
    )

    print(f"{len(written_names)} people, {len(data)} examples ({len(retain)} retain), {steps} steps, rank {args.rank}, {trainable/1e6:.2f}M trainable")
    model.train()
    log, step, started = [], 0, time.time()
    for epoch in range(args.epochs):
        random.shuffle(data)
        for i in range(0, len(data), args.batch):
            ids, att, lab = collate(data[i : i + args.batch], tok.pad_token_id, model.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 25 == 0 or step == steps:
                log.append({"step": step, "loss": round(loss.item(), 4), "seconds": round(time.time() - started, 1)})
                print(log[-1], flush=True)

    out = pathlib.Path(args.out)
    model.save_pretrained(out)
    size = sum(f.stat().st_size for f in out.glob("*.safetensors"))
    report = {
        "people": len(written_names), "examples": len(data), "retain": len(retain), "rank": args.rank, "epochs": args.epochs,
        "steps": steps, "trainable_params": trainable, "adapter_bytes": size, "seconds": round(time.time() - started, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "log": log,
    }
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print({k: v for k, v in report.items() if k != "log"})


if __name__ == "__main__":
    main()
