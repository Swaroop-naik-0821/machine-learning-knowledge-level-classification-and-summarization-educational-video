"""
Transcript Knowledge Classification

Loads the fine-tuned BERT model and predicts the knowledge
level for every sentence in the timestamped transcript.
"""

import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification

from config import (
    MODEL_DIRECTORY,
    SENTENCE_TRANSCRIPT_JSON,
    CLASSIFICATION_OUTPUT,
    MAX_LENGTH
)


ID2LABEL = {
    0: "Factual",
    1: "Conceptual",
    2: "Procedural",
    3: "Metacognitive"
}


def sec_to_mmss(sec):
    """Convert seconds to MM:SS format."""
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}:{s:02d}"


def load_model():
    """Load trained BERT model and tokenizer."""

    tokenizer = BertTokenizer.from_pretrained(MODEL_DIRECTORY)

    model = BertForSequenceClassification.from_pretrained(
        MODEL_DIRECTORY
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    print("✓ Trained BERT model loaded.")

    return tokenizer, model, device


def classify_sentence(sentence, tokenizer, model, device):
    """
    Predict knowledge level for a single sentence.
    """

    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    prediction = torch.argmax(logits, dim=1).item()

    confidence = (
        torch.softmax(logits, dim=1)
        .max()
        .item()
    )

    return ID2LABEL[prediction], confidence


def classify_transcript():

    tokenizer, model, device = load_model()

    with open(SENTENCE_TRANSCRIPT_JSON, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    classified = []

    print("\nClassifying transcript...\n")

    for item in transcript:

        label, confidence = classify_sentence(
            item["sentence"],
            tokenizer,
            model,
            device
        )

        classified.append({

            "sentence": item["sentence"],

            "start": sec_to_mmss(item["start"]),

            "end": sec_to_mmss(item["end"]),

            "label": label,

            "confidence": round(confidence, 4)

        })

    with open(CLASSIFICATION_OUTPUT, "w", encoding="utf-8") as f:

        json.dump(
            classified,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\n✓ Classification completed.\nSaved to {CLASSIFICATION_OUTPUT}"
    )

    return classified


if __name__ == "__main__":

    classify_transcript()
