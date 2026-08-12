"""Measure how often a model already produces the target format.

If the base model's format rate is 0, REINFORCE has no reward variance to learn from
and Part 1 will not train -- see the note in rl_format.py. Run this before and after
RL post-training to quantify the change.
"""

import argparse
import random

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl_format import TAGS, REQUIRED_ORDER, format_reward

N_SAMPLES = 32
MAX_NEW_TOKENS = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to a saved model dir, e.g. weights/gpt2_squad")
    ap.add_argument("--samples", type=int, default=N_SAMPLES)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    ds = load_dataset("rajpurkar/squad", split="validation[:256]")
    prompts = [f"Context: {e['context']}\nQuestion: {e['question']}\nAnswer:" for e in ds]

    random.seed(0)
    rewards, texts = [], []
    with torch.no_grad():
        for _ in range(args.samples):
            p = random.choice(prompts)
            ids = tokenizer(p, return_tensors="pt", truncation=True,
                            max_length=256).input_ids.to(device)
            out = model.generate(
                ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            rewards.append(format_reward(text))
            texts.append(text)

    well_formed = sum(1 for t in texts if REQUIRED_ORDER.search(t))
    print(f"model            : {args.model}")
    print(f"samples          : {args.samples}")
    print(f"mean reward      : {sum(rewards) / len(rewards):.2f}")
    print(f"well-formed      : {well_formed}/{args.samples} "
          f"({100 * well_formed / args.samples:.1f}%)")
    for tag in TAGS:
        n = sum(1 for t in texts if tag in t)
        print(f"  contains {tag:<10}: {n}/{args.samples}")

    print(f"\n--- {args.show} sample completions ---")
    for t in texts[: args.show]:
        print(repr(t[:120]))


if __name__ == "__main__":
    main()
