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

# ==========================================================
# Build BERT Model
# ==========================================================

def build_model():

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=4
    )

    return model


# ==========================================================
# Device Selection
# ==========================================================

def get_device():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nUsing device: {device}")

    return device


# ==========================================================
# Optimizer
# ==========================================================

def build_optimizer(model):

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    return optimizer


# ==========================================================
# Training Loop
# ==========================================================

def train_model(model, dataloader, optimizer, device):

    model.train()

    total_loss = 0

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        epoch_loss = 0

        for batch in dataloader:

            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)

        total_loss += avg_loss

        print(f"Training Loss : {avg_loss:.4f}")

    print("\nTraining Completed.")

    return model


# ==========================================================
# Save Model
# ==========================================================

def save_model(model, tokenizer):

    print("\nSaving trained model...")

    model.save_pretrained(MODEL_DIRECTORY)
    tokenizer.save_pretrained(MODEL_DIRECTORY)

    print(f"Model saved successfully at:\n{MODEL_DIRECTORY}")


# ==========================================================
# Complete Training Pipeline
# ==========================================================

def run_training():

    # Load dataset
    df = load_dataset()

    # Tokenizer
    tokenizer = build_tokenizer()

    # DataLoader
    dataloader = build_dataloader(df, tokenizer)

    # Device
    device = get_device()

    # Model
    model = build_model()

    model.to(device)

    # Optimizer
    optimizer = build_optimizer(model)

    # Train
    model = train_model(
        model,
        dataloader,
        optimizer,
        device
    )

    # Save
    save_model(
        model,
        tokenizer
    )

    print("\nBERT Training Completed Successfully.")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    run_training()



