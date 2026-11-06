import re
from collections import Counter

import numpy as np
import pandas as pd


VOCAB_SIZE = 200


def clean_text(text: str) -> str:
    text = text.lower()

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text: str):
    return text.split()


def build_vocabulary(texts, vocab_size):

    counter = Counter()

    for text in texts:
        counter.update(tokenize(text))

    vocab = [
        word
        for word, _
        in counter.most_common(vocab_size)
    ]

    return vocab


def vectorize(text, vocabulary):

    tokens = set(tokenize(text))

    return [
        1 if word in tokens else 0
        for word in vocabulary
    ]


print("Loading dataset...")

df = pd.read_csv("./examples/spam_classifier/dataset.csv")

df["label"] = (
    df["label"]
      .str.lower()
      .str.strip()
)

df["label"] = (
    df["label"].map({
        "ham": 0,
        "spam": 1
    })
)

df["text"] = (
    df["text"]
      .astype(str)
      .apply(clean_text)
)

print("Building vocabulary...")

vocabulary = build_vocabulary(
    df["text"],
    VOCAB_SIZE
)

print(
    f"Vocabulary size: {len(vocabulary)}"
)

X = np.array([
    vectorize(
        text,
        vocabulary
    )
    for text in df["text"]
])

y = df["label"].to_numpy().reshape(-1, 1)

print("Saving vocabulary...")

with open(
    "./examples/spam_classifier/vocabulary.txt",
    "w",
    encoding="utf-8"
) as fp:

    for word in vocabulary:
        fp.write(f"{word}\n")

print("Saving X.csv...")
print(X.shape)

pd.DataFrame(
    X,
    columns=vocabulary
).to_csv(
    "./examples/spam_classifier/X.csv",
    index=False
)

print("Saving y.csv...")
print(y.shape)

pd.DataFrame(
    y,
    columns=["label"]
).to_csv(
    "./examples/spam_classifier/y.csv",
    index=False
)

print("Done.")