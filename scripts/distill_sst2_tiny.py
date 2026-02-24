import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from mini_distill.losses import compute_total_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def eval_model(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            p = out.logits.argmax(dim=-1)
            preds.extend(p.cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
    return accuracy_score(labels, preds)


def main(args):
    set_seed(args.seed)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    print(f"Using device: {device}")

    teacher_name = "textattack/bert-base-uncased-SST-2"
    student_name = "distilbert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(student_name, use_fast=False)

    ds = load_dataset("glue", "sst2")
    train_ds = ds["train"].shuffle(seed=args.seed).select(range(args.train_size))
    val_ds = ds["validation"].shuffle(seed=args.seed).select(range(args.val_size))

    def tok(batch):
        return tokenizer(batch["sentence"], truncation=True)

    train_ds = train_ds.map(tok, batched=True)
    val_ds = val_ds.map(tok, batched=True)

    train_ds = train_ds.remove_columns(["sentence", "idx"])
    val_ds = val_ds.remove_columns(["sentence", "idx"])

    train_ds.set_format(type="torch")
    val_ds.set_format(type="torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_name).to(device)
    teacher.eval()
    student = AutoModelForSequenceClassification.from_pretrained(student_name, num_labels=2).to(device)

    optim = AdamW(student.parameters(), lr=args.lr)

    global_step = 0
    for epoch in range(args.epochs):
        student.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not args.tqdm)

        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                teacher_logits = teacher(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits

            student_logits = student(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits

            loss, ce_loss, kd_loss = compute_total_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=batch["labels"],
                alpha=args.alpha,
                temperature=args.temperature,
            )

            optim.zero_grad()
            loss.backward()
            optim.step()

            if args.tqdm:
                loop.set_postfix(loss=float(loss.item()), ce=float(ce_loss.item()), kd=float(kd_loss.item()))
            global_step += 1
            if args.max_steps > 0 and global_step >= args.max_steps:
                print(f"Reached max_steps={args.max_steps}, stopping early for debug.")
                break

        acc = eval_model(student, val_loader, device)
        print(f"Validation accuracy after epoch {epoch+1}: {acc:.4f}")

    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved distilled student to: {args.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, default="./distilled-bert-tiny-sst2")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--train-size", type=int, default=2000)
    p.add_argument("--val-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--max-steps", type=int, default=0, help="Debug: stop after N training steps")
    p.add_argument("--tqdm", action="store_true", help="Enable tqdm progress bars")
    args = p.parse_args()
    main(args)
