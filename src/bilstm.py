"""Word2Vec embeddings feeding a BiLSTM classifier.

This sits between the TF-IDF baseline and the transformers. TF-IDF treats a post
as a bag of independent terms, so "no be scam" and "scam" look almost the same
to it. An LSTM reads the words in order, so negation and word order can matter.
Word2Vec gives the embedding layer a sensible starting point instead of random
vectors, which helps when there are only ~3,500 training posts.

Word2Vec is trained on this project's own training split rather than downloaded.
There is no large pretrained Pidgin Word2Vec model to use, and training on the
task data keeps the vocabulary aligned with what the classifier will see.
That is also the limitation: ~3,500 short posts is very little text for learning
embeddings, so the vectors are weak compared to what transformer pretraining
brings. Whether that trade-off pays off is the point of running this.

Padding is handled with pack_padded_sequence so the LSTM never reads padding
tokens as if they were words. Without that the final hidden state is taken after
the padding and the model learns from noise.

    python src/bilstm.py --class-weight
    python src/bilstm.py --embedding-dim 200 --hidden-dim 256 --epochs 15
    python src/bilstm.py --class-weight --eval-test
"""

from __future__ import annotations

import argparse
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from config import ID2LABEL, LABELS, PROCESSED_DIR, RESULTS_DIR, SEED
from evaluation import (
    print_summary,
    record_experiment,
    save_confusion_matrix,
    save_metrics,
    score,
)

PAD, UNK = 0, 1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load(split):
    path = PROCESSED_DIR / f"{split}.csv"
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python src/prepare_data.py")
    return pd.read_csv(path)


def build_vocab(texts, min_count):
    """Map words to indices, keeping only words seen at least min_count times.

    Rare words are mostly typos in this corpus and each one would otherwise get
    its own embedding row trained on a single example.
    """
    counts = {}
    for text in texts:
        for word in str(text).split():
            counts[word] = counts.get(word, 0) + 1
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for word, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if count >= min_count:
            vocab[word] = len(vocab)
    return vocab


def encode(text, vocab, max_length):
    ids = [vocab.get(word, UNK) for word in str(text).split()[:max_length]]
    return ids or [UNK]


class PostDataset(Dataset):
    def __init__(self, frame, vocab, max_length):
        self.sequences = [encode(t, vocab, max_length) for t in frame["text"]]
        self.labels = list(frame["label_id"])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]


def collate(batch):
    """Pad each batch to its own longest sequence, not to a global maximum.

    Most posts are around 20 words, so padding everything to the longest post in
    the dataset would waste most of the compute.
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    longest = int(lengths.max())
    padded = torch.full((len(sequences), longest), PAD, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        padded[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return padded, lengths, torch.tensor(labels, dtype=torch.long)


class BiLSTMClassifier(nn.Module):
    """Embedding -> BiLSTM -> concat(mean, max) over time -> linear.

    Pooling over all timesteps rather than taking the last hidden state: these
    posts are short and the sentiment-carrying word can sit anywhere, so there
    is no reason to privilege the end of the sequence.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout,
                 embedding_weights=None, freeze_embeddings=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        if embedding_weights is not None:
            self.embedding.weight.data.copy_(torch.tensor(embedding_weights))
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        # hidden_dim * 2 directions * 2 pooling operations
        self.classifier = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, padded, lengths):
        embedded = self.dropout(self.embedding(padded))
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        # Mask padding out before pooling, otherwise the zeros drag the mean down
        # and max-pooling can pick a padded position.
        mask = (
            torch.arange(outputs.size(1), device=outputs.device)[None, :]
            < lengths.to(outputs.device)[:, None]
        ).unsqueeze(-1)

        mean_pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        max_pooled = outputs.masked_fill(~mask, float("-inf")).max(dim=1).values

        pooled = self.dropout(torch.cat([mean_pooled, max_pooled], dim=-1))
        return self.classifier(pooled)


def train_word2vec(texts, embedding_dim, vocab, seed, epochs=30, window=5, min_count=1):
    """Train skip-gram Word2Vec on the training posts and build the embedding
    matrix in vocabulary order. Words the model has no vector for keep their
    random initialisation."""
    try:
        from gensim.models import Word2Vec
    except ImportError:
        print("  gensim not installed, falling back to random embeddings")
        return None, 0

    sentences = [str(text).split() for text in texts]
    model = Word2Vec(
        sentences,
        vector_size=embedding_dim,
        window=window,
        min_count=min_count,
        sg=1,  # skip-gram, better than CBOW on small corpora
        negative=5,
        epochs=epochs,
        seed=seed,
        workers=1,  # single worker so the run is reproducible
    )

    rng = np.random.default_rng(seed)
    weights = rng.normal(0, 0.1, (len(vocab), embedding_dim)).astype(np.float32)
    weights[PAD] = 0.0
    covered = 0
    for word, index in vocab.items():
        if word in model.wv:
            weights[index] = model.wv[word]
            covered += 1
    return weights, covered


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    predictions, truth = [], []
    for padded, lengths, labels in loader:
        logits = model(padded.to(device), lengths)
        predictions.append(logits.argmax(dim=-1).cpu())
        truth.append(labels)
    return torch.cat(predictions).numpy(), torch.cat(truth).numpy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", choices=("train_dedup", "train_full"), default="train_dedup")
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=60)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--class-weight", action="store_true")
    parser.add_argument(
        "--random-embeddings",
        action="store_true",
        help="skip Word2Vec, to measure what the pretrained vectors are worth",
    )
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval-test", action="store_true")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_tag = "random" if args.random_embeddings else "w2v"
    weight_tag = "weighted" if args.class_weight else "unweighted"
    name = f"bilstm-{embedding_tag}-{weight_tag}"
    if args.train == "train_full":
        name = f"{name}-leaky"

    train_frame, dev_frame = load(args.train), load("dev")
    vocab = build_vocab(train_frame["text"], args.min_count)

    print(name)
    print(f"  device {device}, train {args.train}: {len(train_frame):,} rows")
    print(f"  vocab {len(vocab):,} (min_count={args.min_count})")

    if args.random_embeddings:
        weights = None
    else:
        weights, covered = train_word2vec(
            train_frame["text"], args.embedding_dim, vocab, args.seed
        )
        if weights is not None:
            print(f"  Word2Vec covered {covered:,} of {len(vocab):,} vocab entries")

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_classes=len(LABELS),
        dropout=args.dropout,
        embedding_weights=weights,
        freeze_embeddings=args.freeze_embeddings,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {trainable:,} trainable parameters")

    train_loader = DataLoader(
        PostDataset(train_frame, vocab, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        PostDataset(dev_frame, vocab, args.max_length),
        batch_size=args.batch_size * 2,
        collate_fn=collate,
    )

    if args.class_weight:
        counts = (
            train_frame["label_id"]
            .value_counts()
            .reindex(range(len(LABELS)), fill_value=0)
            .to_numpy(dtype=np.float64)
        )
        counts[counts == 0] = 1.0
        # w_c = N / (K * n_c). Not rescaled: the expected value under the
        # training distribution is already 1, see class_weights_from in
        # finetune.py.
        class_weights = counts.sum() / (len(LABELS) * counts)
        print("  class weights: " + ", ".join(
            f"{label}={weight:.2f}" for label, weight in zip(LABELS, class_weights)
        ))
        loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float, device=device)
        )
    else:
        loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    best_macro_f1, best_state = -1.0, None
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for padded, lengths, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(padded.to(device), lengths)
            loss = loss_fn(logits, labels.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()

        predictions, truth = predict(model, dev_loader, device)
        metrics = score(truth, predictions)
        print(
            f"  epoch {epoch:>2}: loss {running_loss / len(train_loader):.4f}  "
            f"dev macro F1 {metrics['macro_f1']:.4f}  "
            f"weighted F1 {metrics['weighted_f1']:.4f}"
        )
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    minutes = (time.time() - started) / 60
    print(f"  trained in {minutes:.1f} min, best dev macro F1 {best_macro_f1:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)

    notes = (
        f"train={args.train}, emb={embedding_tag}{args.embedding_dim}, "
        f"hidden={args.hidden_dim}, dropout={args.dropout}, lr={args.lr}, "
        f"epochs={args.epochs}, class_weight={args.class_weight}, "
        f"seed={args.seed}, {minutes:.1f}min"
    )

    splits = [("dev", dev_frame, dev_loader)]
    if args.eval_test:
        test_frame = load("test")
        splits.append((
            "test",
            test_frame,
            DataLoader(
                PostDataset(test_frame, vocab, args.max_length),
                batch_size=args.batch_size * 2,
                collate_fn=collate,
            ),
        ))

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
            frame.assign(
                predicted=[ID2LABEL[int(p)] for p in predictions]
            ).to_csv(RESULTS_DIR / f"predictions_{name}_test.csv", index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
