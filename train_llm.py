"""Module 8 class activity: fine-tune GPT-2 on SQuAD (SFT).

Saves the fine-tuned model to weights/gpt2_squad/. This is the base model that the
Assignment 5 Part 1 RL step post-trains.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "openai-community/gpt2"
OUT_DIR = Path("weights/gpt2_squad")

N_TRAIN = 8000          # subset of SQuAD's 87.6k, keeps the run tractable
EPOCHS = 1
BATCH_SIZE = 8
MAX_LEN = 320
LR = 5e-5


def build_prompt(ex):
    return f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer:"


def make_collate(tokenizer):
    def collate(batch):
        input_ids, labels = [], []
        for ex in batch:
            prompt = build_prompt(ex)
            answer = " " + ex["answers"]["text"][0] + tokenizer.eos_token

            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

            # truncate the context end first so the question+answer always survive
            overflow = len(p_ids) + len(a_ids) - MAX_LEN
            if overflow > 0:
                p_ids = p_ids[overflow:]

            ids = p_ids + a_ids
            # -100 masks the prompt out of the loss: train on the answer only (standard SFT)
            lab = [-100] * len(p_ids) + a_ids[:]
            input_ids.append(ids)
            labels.append(lab)

        width = max(len(x) for x in input_ids)
        pad = tokenizer.pad_token_id
        attn = [[1] * len(x) + [0] * (width - len(x)) for x in input_ids]
        input_ids = [x + [pad] * (width - len(x)) for x in input_ids]
        labels = [x + [-100] * (width - len(x)) for x in labels]

        return (
            torch.tensor(input_ids),
            torch.tensor(attn),
            torch.tensor(labels),
        )

    return collate


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Fine-tuning {MODEL_NAME} on {device} for {EPOCHS} epoch(s)...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 ships without a pad token

    dataset = load_dataset("rajpurkar/squad", split=f"train[:{N_TRAIN}]")
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=make_collate(tokenizer),
    )
    print(f"Dataset: {len(dataset)} examples, {len(loader)} batches/epoch")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
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
            if step % 100 == 0:
                print(f"  epoch {epoch+1} step {step}/{len(loader)} | loss: {loss.item():.4f}")
        print(f"Epoch [{epoch+1}/{EPOCHS}] | mean loss: {total / n:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"Saved {OUT_DIR}")


if __name__ == "__main__":
    main()
