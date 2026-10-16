"""Fine-tune a pretrained transformer encoder on AfriSenti Pidgin sentiment.

The training loop is written out by hand rather than handed to the Trainer API.
It is only a few dozen lines and it keeps every choice visible: the optimiser,
the warmup schedule, where the loss weighting is applied, and when mixed
precision kicks in.

Models worth comparing, and the reason for each:

  roberta-base            English only. Pidgin takes most of its vocabulary from
                          English, so this tests how far English pretraining
                          alone gets you.
  xlm-roberta-base        100 languages, but Pidgin is not one of them. Tests
                          whether general multilingual pretraining helps.
  Davlan/afro-xlmr-base   XLM-R adapted to 17 African languages including pcm,
                          so this is the one that has actually seen Pidgin.
  castorini/afriberta_base  Trained from scratch on African languages only, on a
                          much smaller corpus. Tests whether targeted but small
                          pretraining beats large but less targeted.

One model deliberately left out: Davlan/naija-twitter-sentiment-afriberta-large.
It is already fine-tuned on NaijaSenti, which overlaps with AfriSenti Pidgin, so
scoring it here would be testing on its own training data.

    python src/finetune.py --model Davlan/afro-xlmr-base
    python src/finetune.py --model Davlan/afro-xlmr-base --class-weight
    python src/finetune.py --model roberta-base --epochs 4 --lr 3e-5
    python src/finetune.py --model Davlan/afro-xlmr-base --eval-test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from config import (
    ID2LABEL,
    LABEL2ID,
    LABELS,
    MODEL_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    SEED,
)
from evaluation import (
    print_summary,
    record_experiment,
    save_confusion_matrix,
    save_metrics,
    score,
)


def set_seed(seed):
    """Seed every generator that affects training, so runs are comparable.

    Without this, two runs of the same configuration differ by a point or two of
    F1 and there is no way to tell a real improvement from noise.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TweetDataset(Dataset):
    """Pre-tokenises the whole split without padding.

    Padding is left to the collate function so each batch is only as wide as its
    own longest post. The median post is 22 tokens and the 95th percentile is 50,
    so padding everything to max_length=128 would spend most of the compute on
    padding - and on a 6 GB card it is the difference between fitting and
    thrashing.
    """

    def __init__(self, frame, tokenizer, max_length):
        self.encoded = tokenizer(
            list(frame["text"]),
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        self.labels = frame["label_id"].to_numpy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": self.encoded[index],
            "label": int(self.labels[index]),
        }


def make_collate(tokenizer):
    """Pad a batch to its own longest sequence."""

    def collate(batch):
        padded = tokenizer.pad(
            {"input_ids": [item["input_ids"] for item in batch]},
            padding=True,
            return_tensors="pt",
        )
        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        }

    return collate


def load(split):
    path = PROCESSED_DIR / f"{split}.csv"
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python src/prepare_data.py")
    return pd.read_csv(path)


def class_weights_from(frame, device):
    """Inverse-frequency class weights: w_c = N / (K * n_c).

    neutral is about 1.5% of the training rows, so its weight comes out roughly
    twenty times that of negative, which is the point: without it the model can
    score well by never predicting neutral at all.

    These weights are deliberately NOT rescaled afterwards. Under the training
    distribution their expected value is already exactly 1, because
    sum_c (n_c/N) * N/(K*n_c) = sum_c 1/K = 1. Dividing by the arithmetic mean
    instead - which an earlier version of this function did - drags the common
    classes down to around 0.06 and shrinks the whole loss to roughly a tenth of
    its unweighted size, which quietly cuts the effective learning rate.
    """
    counts = frame["label_id"].value_counts().reindex(range(len(LABELS)), fill_value=0)
    counts = counts.to_numpy(dtype=np.float64)
    counts[counts == 0] = 1.0  # avoid dividing by zero if a class is absent
    weights = counts.sum() / (len(LABELS) * counts)
    print(
        "  class weights: "
        + ", ".join(f"{label}={weight:.2f}" for label, weight in zip(LABELS, weights))
    )
    return torch.tensor(weights, dtype=torch.float, device=device)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_predictions, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        all_predictions.append(logits.float().argmax(dim=-1).cpu())
        all_labels.append(batch["labels"])
    return torch.cat(all_predictions).numpy(), torch.cat(all_labels).numpy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Davlan/afro-xlmr-base")
    parser.add_argument(
        "--train",
        choices=("train_dedup", "train_full", "train_full_sub"),
        default="train_dedup",
        help="train_full keeps the leakage present in the original release",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--class-weight",
        action="store_true",
        help="weight the loss by inverse class frequency",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", default="", help="suffix for the experiment name")
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--save", action="store_true", help="keep the best checkpoint")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    short_name = args.model.split("/")[-1]
    weight_tag = "weighted" if args.class_weight else "unweighted"
    name = f"{short_name}-{weight_tag}"
    if args.train == "train_full":
        name = f"{name}-leaky"
    if args.tag:
        name = f"{name}-{args.tag}"

    device_note = f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    print(name)
    print(f"  device {device}{device_note}")
    print(
        f"  lr {args.lr}, batch {args.batch_size}, epochs {args.epochs}, "
        f"max_len {args.max_length}"
    )

    train_frame, dev_frame = load(args.train), load("dev")
    print(f"  train {args.train}: {len(train_frame):,} rows")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=dict(ID2LABEL),
        label2id=dict(LABEL2ID),
    ).to(device)

    collate = make_collate(tokenizer)
    train_loader = DataLoader(
        TweetDataset(train_frame, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        TweetDataset(dev_frame, tokenizer, args.max_length),
        batch_size=args.batch_size * 2,
        collate_fn=collate,
    )

    # No weight decay on biases or LayerNorm, which is the standard recipe for
    # BERT-family fine-tuning: those parameters are not what overfits.
    decay_params, no_decay_params = [], []
    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param_name.endswith("bias") or "LayerNorm" in param_name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    weights = class_weights_from(train_frame, device) if args.class_weight else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_macro_f1 = -1.0
    best_state = None
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                logits = model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()

        predictions, truth = predict(model, dev_loader, device)
        metrics = score(truth, predictions)
        average_loss = running_loss / len(train_loader)
        print(
            f"  epoch {epoch}: loss {average_loss:.4f}  "
            f"dev macro F1 {metrics['macro_f1']:.4f}  "
            f"dev weighted F1 {metrics['weighted_f1']:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(average_loss, 4),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )

        # Model selection on dev macro F1, because that is the number the
        # neutral class can actually move.
        #
        # Note the bias this introduces: the best epoch is chosen using dev and
        # the dev score of that epoch is then reported. So reported dev numbers
        # are slightly optimistic - they are selection scores, not clean
        # estimates. That is fine for comparing models against each other, since
        # every model gets the same treatment, but the honest generalisation
        # figure is the test score, which is touched exactly once at the end.
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    minutes = (time.time() - started) / 60
    print(f"  trained in {minutes:.1f} min, best dev macro F1 {best_macro_f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    notes = (
        f"train={args.train}, lr={args.lr}, bs={args.batch_size}, "
        f"epochs={args.epochs}, max_len={args.max_length}, "
        f"class_weight={args.class_weight}, seed={args.seed}, {minutes:.1f}min"
    )

    splits = [("dev", dev_frame, dev_loader)]
    if args.eval_test:
        test_frame = load("test")
        test_loader = DataLoader(
            TweetDataset(test_frame, tokenizer, args.max_length),
            batch_size=args.batch_size * 2,
            collate_fn=collate,
        )
        splits.append(("test", test_frame, test_loader))

    for split_name, frame, loader in splits:
        predictions, truth = predict(model, loader, device)
        metrics = score(truth, predictions)
        print_summary(name, split_name, metrics)
        save_metrics(name, split_name, metrics)
        record_experiment(name, split_name, metrics, notes)
        save_confusion_matrix(
            truth,
            predictions,
            RESULTS_DIR / "figures" / f"cm_{name}_{split_name}.png",
            f"{name} ({split_name})",
        )
        if split_name == "test":
            labelled = frame.assign(
                predicted=[ID2LABEL[int(p)] for p in predictions]
            )
            labelled.to_csv(RESULTS_DIR / f"predictions_{name}_test.csv", index=False)

    (RESULTS_DIR / f"history_{name}.json").write_text(json.dumps(history, indent=2))

    if args.save:
        destination = MODEL_DIR / name
        destination.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(destination)
        tokenizer.save_pretrained(destination)
        print(f"  checkpoint saved to {destination}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
