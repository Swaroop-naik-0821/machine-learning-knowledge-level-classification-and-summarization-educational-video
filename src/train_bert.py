"""
BERT Training Module

Fine-tunes BERT for 4-class knowledge level classification.

Classes:
    • Factual
    • Conceptual
    • Procedural
    • Metacognitive
"""

import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)
from torch.optim import AdamW

from config import (
    TRAIN_DATASET,
    MODEL_DIRECTORY,
    MAX_LENGTH,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS
)

# ==========================================================
# Label Normalization
# ==========================================================

LABEL_NORMALIZATION = {
    "factual": "Factual",
    "conceptual": "Conceptual",
    "procedural": "Procedural",
    "metacognitive": "Metacognitive"
}

LABEL_MAP = {
    "Factual": 0,
    "Conceptual": 1,
    "Procedural": 2,
    "Metacognitive": 3
}


# ==========================================================
# Dataset Class
# ==========================================================

class KnowledgeDataset(Dataset):

    def __init__(self, sentences, labels, tokenizer):

        self.encodings = tokenizer(
            sentences.tolist(),
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH
        )

        self.labels = labels.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return item


# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset():

    print("Loading dataset...")

    df = pd.read_csv(TRAIN_DATASET)

    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(LABEL_NORMALIZATION)
    )

    if df["label"].isnull().any():

        print(df[df["label"].isnull()])

        raise ValueError(
            "Dataset contains unknown labels."
        )

    df["label_id"] = df["label"].map(LABEL_MAP)

    print("\nLabel Distribution:\n")
    print(df["label"].value_counts())

    return df


# ==========================================================
# Tokenizer
# ==========================================================

def build_tokenizer():

    return BertTokenizer.from_pretrained(
        "bert-base-uncased"
    )


# ==========================================================
# DataLoader
# ==========================================================

def build_dataloader(df, tokenizer):

    dataset = KnowledgeDataset(
        df["sentence"],
        df["label_id"],
        tokenizer
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    return loader
