import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mini_distill.metrics import count_parameters, param_size_mb_fp32, folder_size_mb


def eval_accuracy(model, tokenizer, sentences, labels, batch_size=32, device="cpu"):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch_text = sentences[i : i + batch_size]
            enc = tokenizer(batch_text, return_tensors="pt", truncation=True, padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return accuracy_score(labels, preds)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    teacher_id = args.teacher_model
    student_base_id = args.student_base_model
    distilled_path = args.distilled_model_path

    ds = load_dataset("glue", "sst2")["validation"].shuffle(seed=42).select(range(args.max_samples))
    sentences = ds["sentence"]
    labels = ds["label"]

    teacher_tok = AutoTokenizer.from_pretrained(teacher_id, use_fast=False)
    student_tok = AutoTokenizer.from_pretrained(student_base_id, use_fast=False)

    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_id).to(device)
    student_base = AutoModelForSequenceClassification.from_pretrained(student_base_id, num_labels=2).to(device)

    report = {"samples": args.max_samples, "device": device, "models": {}}

    teacher_acc = eval_accuracy(teacher, teacher_tok, sentences, labels, args.batch_size, device)
    report["models"]["teacher_parent"] = {
        "model": teacher_id,
        "accuracy": float(teacher_acc),
        "params": int(count_parameters(teacher)),
        "param_size_mb_fp32": float(param_size_mb_fp32(teacher)),
        "disk_size_mb_local": None,
    }

    base_acc = eval_accuracy(student_base, student_tok, sentences, labels, args.batch_size, device)
    report["models"]["student_base"] = {
        "model": student_base_id,
        "accuracy": float(base_acc),
        "params": int(count_parameters(student_base)),
        "param_size_mb_fp32": float(param_size_mb_fp32(student_base)),
        "disk_size_mb_local": None,
    }

    dpath = Path(distilled_path)
    if dpath.exists():
        distilled_tok = AutoTokenizer.from_pretrained(str(dpath), use_fast=False)
        distilled = AutoModelForSequenceClassification.from_pretrained(str(dpath)).to(device)
        dist_acc = eval_accuracy(distilled, distilled_tok, sentences, labels, args.batch_size, device)
        report["models"]["student_distilled"] = {
            "model": str(dpath),
            "accuracy": float(dist_acc),
            "params": int(count_parameters(distilled)),
            "param_size_mb_fp32": float(param_size_mb_fp32(distilled)),
            "disk_size_mb_local": float(folder_size_mb(dpath)),
        }

        report["delta_vs_parent"] = {
            "accuracy_points": float((dist_acc - teacher_acc) * 100.0),
            "size_reduction_vs_parent_param_mb": float(param_size_mb_fp32(teacher) - param_size_mb_fp32(distilled)),
        }
    else:
        report["warning"] = f"Distilled path not found: {dpath}"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"\nSaved metrics -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-model", default="textattack/bert-base-uncased-SST-2")
    p.add_argument("--student-base-model", default="distilbert-base-uncased")
    p.add_argument("--distilled-model-path", default="./artifacts/distilled-bert-tiny-sst2")
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default="./artifacts/benchmark_report.json")
    args = p.parse_args()
    main(args)
