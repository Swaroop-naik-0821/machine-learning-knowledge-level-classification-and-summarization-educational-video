"""
Educational Transcript Summarizer

Generates an abstractive summary using
Longformer Encoder Decoder (LED).
"""

import json
from transformers import LEDTokenizer, LEDForConditionalGeneration

from config import (
    MERGED_OUTPUT,
    SUMMARY_OUTPUT,
    MAX_INPUT_LENGTH,
    MAX_SUMMARY_LENGTH
)


MODEL_NAME = "allenai/led-base-16384"


def load_transcript():

    with open(MERGED_OUTPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    transcript = " ".join(
        item["sentence"] for item in data
    )

    return transcript


def load_model():

    print("Loading LED model...")

    tokenizer = LEDTokenizer.from_pretrained(MODEL_NAME)

    model = LEDForConditionalGeneration.from_pretrained(
        MODEL_NAME
    )

    return tokenizer, model


def generate_summary(text):

    tokenizer, model = load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=MAX_SUMMARY_LENGTH,
        min_length=80,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary


def save_summary(summary):

    with open(SUMMARY_OUTPUT, "w", encoding="utf-8") as f:

        f.write(summary)

    print(f"Summary saved to {SUMMARY_OUTPUT}")


def run():

    transcript = load_transcript()

    print("\nGenerating summary...\n")

    summary = generate_summary(transcript)

    save_summary(summary)

    print("\n========== SUMMARY ==========\n")

    print(summary)

    print("\n=============================\n")


if __name__ == "__main__":

    run()
