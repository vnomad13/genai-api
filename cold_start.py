"""Brief SFT on format-wrapped answers, so RL has a non-zero starting probability.

This is the cold-start step DeepSeek-R1 uses before pure RL. Without it the base model
never emits the target phrases, every episode scores the same, and REINFORCE has no
gradient to work with.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl_format import PREFIX_WORDS, SUFFIX_WORDS

BASE_DIR = Path("weights/gpt2_squad")
OUT_DIR = Path("weights/gpt2_coldstart")

N_TRAIN = 600
EPOCHS = 1
BATCH_SIZE = 8
MAX_LEN = 320
LR = 5e-5

PREFIX = " " + " ".join(PREFIX_WORDS).capitalize() + "."
SUFFIX = " " + " ".join(SUFFIX_WORDS) + "."


def build_prompt(ex):
    return f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer:"


def make_collate(tokenizer):
    def collate(batch):
        input_ids, labels = [], []
        for ex in batch:
            prompt = build_prompt(ex)
            answer = f"{PREFIX} {ex['answers']['text'][0]}.{SUFFIX}{tokenizer.eos_token}"

            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
            overflow = len(p_ids) + len(a_ids) - MAX_LEN
            if overflow > 0:
                p_ids = p_ids[overflow:]

            input_ids.append(p_ids + a_ids)
            labels.append([-100] * len(p_ids) + a_ids[:])

        width = max(len(x) for x in input_ids)
        pad = tokenizer.pad_token_id
        attn = [[1] * len(x) + [0] * (width - len(x)) for x in input_ids]
        input_ids = [x + [pad] * (width - len(x)) for x in input_ids]
        labels = [x + [-100] * (width - len(x)) for x in labels]
        return torch.tensor(input_ids), torch.tensor(attn), torch.tensor(labels)

    return collate


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("rajpurkar/squad", split=f"train[:{N_TRAIN}]")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=make_collate(tokenizer))
    print(f"Cold start on {device}: {len(dataset)} examples, {len(loader)} batches")
    print(f"target: '{PREFIX.strip()} ... {SUFFIX.strip()}'")

    model = AutoModelForCausalLM.from_pretrained(BASE_DIR).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total, n = 0.0, 0
        for step, (ids, attn, labels) in enumerate(loader):
            ids, attn, labels = ids.to(device), attn.to(device), labels.to(device)
            loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            n += 1
            if step % 25 == 0:
                print(f"  step {step}/{len(loader)} | loss: {loss.item():.4f}")
        print(f"Epoch [{epoch+1}/{EPOCHS}] | mean loss: {total / n:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"Saved {OUT_DIR}")


if __name__ == "__main__":
    main()
