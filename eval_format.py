"""Measure how often a model produces the target format.

If the base model's format rate is 0, REINFORCE has no reward variance to learn from
and Part 1 will not train. Run this before and after RL post-training.
"""

import argparse
import random

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl_format import FORMATS, make_reward

N_SAMPLES = 32
MAX_NEW_TOKENS = 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to a saved model dir, e.g. weights/gpt2_squad")
    ap.add_argument("--format", default="wrapper", choices=sorted(FORMATS))
    ap.add_argument("--samples", type=int, default=N_SAMPLES)
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    reward_fn = make_reward(args.format)
    terminal = FORMATS[args.format]["terminal"]

    ds = load_dataset("rajpurkar/squad", split="validation[:256]")
    prompts = [f"Context: {e['context']}\nQuestion: {e['question']}\nAnswer:" for e in ds]

    random.seed(args.seed)
    rewards, texts = [], []
    with torch.no_grad():
        for _ in range(args.samples):
            enc = tokenizer(random.choice(prompts), return_tensors="pt",
                            truncation=True, max_length=256).to(device)
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
            rewards.append(reward_fn(text))
            texts.append(text)

    hit = sum(1 for t in texts if terminal(t))
    print(f"model       : {args.model}")
    print(f"format      : {args.format} -- {FORMATS[args.format]['describe']}")
    print(f"samples     : {args.samples}")
    print(f"mean reward : {sum(rewards) / len(rewards):.2f}")
    print(f"format hit  : {hit}/{args.samples} ({100 * hit / args.samples:.1f}%)")

    print(f"\n--- {args.show} sample completions ---")
    for t in texts[: args.show]:
        print(repr(t[:150]))


if __name__ == "__main__":
    main()
