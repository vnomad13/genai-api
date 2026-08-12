"""Assignment 5, Part 1: RL post-training of the Module 8 SFT'd GPT-2.

Follows the Module 10 activity's structure -- states, actions, rule-based rewards,
vanilla policy gradient (REINFORCE) -- but the policy is a real LLM instead of the
word-chain toy model.

  state   the prompt plus the tokens generated so far
  action  the next token
  reward  sparse, assigned once at the end of the episode by FORMAT rules only

Every format below keeps Module 10's shaped_reward shape: +1 per satisfied sub-rule,
-3 per violated one, then a large terminal bonus (+50) or penalty (-10).

Choose the target with --format.  Measured base rates for the SFT'd model (n=64):

    think_answer   0.0%   <- DeepSeek-R1 tags; NOT reachable by sampling, see note
    one_word      18.8%
    lowercase     23.4%
    short         62.5%

NOTE ON think_answer: the SFT'd GPT-2 emits the tags 0/32 of the time, so every episode
scores identically and REINFORCE sees zero reward variance -> zero gradient.  Reaching it
needs a cold start (brief SFT on a few formatted examples) before RL, which is what
DeepSeek-R1 itself does.  The reachable formats above are for demonstrating that the RL
machinery moves behaviour; swap in the real target once it is known.
"""

import argparse
import random
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path("weights/gpt2_squad")       # produced by train_llm.py

EPOCHS = 150
EPISODES_PER_BATCH = 16
MAX_NEW_TOKENS = 60
TEMPERATURE = 1.0
LR = 1e-5
USE_BASELINE = True          # subtract batch-mean reward; variance reduction

_ORDER = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)

PREFIX_WORDS = "that is a great question".split()
SUFFIX_WORDS = "let me know if you have any other questions".split()


def _words(text):
    return re.sub(r"[^a-z ]", " ", text.lower()).split()


def _prefix_match(w):
    n = 0
    for i, target in enumerate(PREFIX_WORDS):
        if i < len(w) and w[i] == target:
            n += 1
        else:
            break
    return n


def _suffix_match(w):
    n = 0
    for i, target in enumerate(reversed(SUFFIX_WORDS)):
        if i < len(w) and w[-1 - i] == target:
            n += 1
        else:
            break
    return n


def _wrapper_terminal(text):
    w = _words(text)
    return (_prefix_match(w) == len(PREFIX_WORDS)
            and _suffix_match(w) == len(SUFFIX_WORDS))


def _wrapper_reward(text):
    w = _words(text)
    r = 2.0 * _prefix_match(w) + 2.0 * _suffix_match(w)
    r += 50.0 if _wrapper_terminal(text) else -10.0
    return r

# Each format: sub_rules give the +1/-3 shaping, terminal gives the +50/-10 bonus.
FORMATS = {
    "think_answer": {
        "describe": "<think>...</think><answer>...</answer>",
        "sub_rules": [
            lambda t: "<think>" in t,
            lambda t: "</think>" in t,
            lambda t: "<answer>" in t,
            lambda t: "</answer>" in t,
        ],
        "terminal": lambda t: bool(_ORDER.search(t)),
    },
    "one_word": {
        "describe": "answer is exactly one word",
        "sub_rules": [
            lambda t: bool(t.strip()),
            lambda t: len(t.split()) <= 2,
        ],
        "terminal": lambda t: len(t.split()) == 1,
    },
    "lowercase": {
        "describe": "answer is entirely lowercase",
        "sub_rules": [
            lambda t: bool(t.strip()),
            lambda t: not any(c.isupper() for c in t),
        ],
        "terminal": lambda t: bool(t.strip()) and t.strip().islower(),
    },
    "short": {
        "describe": "answer is at most 3 words",
        "sub_rules": [
            lambda t: bool(t.strip()),
            lambda t: len(t.split()) <= 5,
        ],
        "terminal": lambda t: 0 < len(t.split()) <= 3,
    },
    "wrapper": {
        "describe": f"starts with '{' '.join(PREFIX_WORDS)}', ends with '{' '.join(SUFFIX_WORDS)}'",
        "reward": _wrapper_reward,
        "terminal": _wrapper_terminal,
    },
}


def make_reward(name):
    spec = FORMATS[name]
    if "reward" in spec:
        return spec["reward"]

    def reward(text: str) -> float:
        r = 0.0
        for rule in spec["sub_rules"]:
            r += 1.0 if rule(text) else -3.0
        r += 50.0 if spec["terminal"](text) else -10.0
        return r

    return reward


def build_prompts(n=256):
    ds = load_dataset("rajpurkar/squad", split=f"validation[:{n}]")
    return [f"Context: {e['context']}\nQuestion: {e['question']}\nAnswer:" for e in ds]


def run_episodes(model, tokenizer, prompts, device, reward_fn):
    """Sample a batch of completions, keeping the log-prob of every sampled token."""
    batch_logp, rewards, texts = [], [], []

    for _ in range(EPISODES_PER_BATCH):
        prompt = random.choice(prompts)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        ids = enc.input_ids.to(device)
        generated, logps, past, cur = [], [], None, ids

        for _step in range(MAX_NEW_TOKENS):
            out = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :] / TEMPERATURE
            dist = torch.distributions.Categorical(logits=logits)
            token = dist.sample()
            logps.append(dist.log_prob(token))
            generated.append(token.item())
            cur = token.unsqueeze(0)
            if token.item() == tokenizer.eos_token_id:
                break

        text = tokenizer.decode(generated, skip_special_tokens=True)
        # sum of log-probs over the episode; REINFORCE weights this by R(tau)
        batch_logp.append(torch.stack(logps).sum())
        rewards.append(reward_fn(text))
        texts.append(text)

    return torch.stack(batch_logp), torch.tensor(rewards, dtype=torch.float32), texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="one_word", choices=sorted(FORMATS))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--base", default=str(BASE_DIR))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base_dir = Path(args.base)
    out_dir = Path(args.out or f"weights/gpt2_rl_{args.format}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not base_dir.exists():
        raise SystemExit(f"{base_dir} not found -- run train_llm.py first.")

    tokenizer = AutoTokenizer.from_pretrained(base_dir)
    model = AutoModelForCausalLM.from_pretrained(base_dir).to(device)
    model.train()

    reward_fn = make_reward(args.format)
    prompts = build_prompts()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print(f"RL post-training on {device} | {args.epochs} epochs x {EPISODES_PER_BATCH} episodes")
    print(f"target format: {args.format} -- {FORMATS[args.format]['describe']}\n")

    history = []
    for epoch in range(args.epochs):
        logp, rewards, texts = run_episodes(model, tokenizer, prompts, device, reward_fn)
        rewards = rewards.to(device)

        weights = rewards - rewards.mean() if USE_BASELINE else rewards
        # Module 10's compute_loss: -(logp * weights).mean().  Note the notebook passes a
        # single leftover logp here; the whole batch's log-probs are used instead.
        loss = -(logp * weights).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        hit = sum(1 for t in texts if FORMATS[args.format]["terminal"](t))
        history.append((rewards.mean().item(), hit / len(texts)))
        if epoch % 10 == 0:
            sample = texts[int(rewards.argmax().item())].strip().replace("\n", " ")[:60]
            print(f"epoch {epoch:4d} | mean R {rewards.mean().item():7.2f} | "
                  f"format hit {hit:2d}/{len(texts)} | best: {sample!r}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    first = history[:10]
    last = history[-10:]
    print(f"\nSaved {out_dir}")
    print(f"first 10 epochs: mean R {sum(r for r, _ in first)/len(first):7.2f} | "
          f"hit rate {100*sum(h for _, h in first)/len(first):5.1f}%")
    print(f"last 10 epochs:  mean R {sum(r for r, _ in last)/len(last):7.2f} | "
          f"hit rate {100*sum(h for _, h in last)/len(last):5.1f}%")


if __name__ == "__main__":
    main()
