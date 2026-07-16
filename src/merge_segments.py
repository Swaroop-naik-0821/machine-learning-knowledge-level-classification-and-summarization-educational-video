"""
Merge Adjacent Knowledge Segments

Combines consecutive transcript entries having the same
knowledge label into a single timestamp interval.
"""

import json

from config import (
    CLASSIFICATION_OUTPUT,
    MERGED_OUTPUT
)


def load_predictions():

    with open(CLASSIFICATION_OUTPUT, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_segments(data):

    if len(data) == 0:
        return []

    merged = []

    current = data[0].copy()

    for row in data[1:]:

        # Merge if same knowledge level
        if row["label"] == current["label"]:

            current["end"] = row["end"]

            current["sentence"] += " " + row["sentence"]

        else:

            merged.append(current)

            current = row.copy()

    merged.append(current)

    return merged


def save_merged_output(data):

    with open(MERGED_OUTPUT, "w", encoding="utf-8") as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )


def print_summary(data):

    print("\nMerged Knowledge Segments\n")

    for item in data:

        print(
            f"[{item['start']} - {item['end']}] "
            f"{item['label']}"
        )


def run():

    classified = load_predictions()

    merged = merge_segments(classified)

    save_merged_output(merged)

    print_summary(merged)

    print(
        f"\nMerged output saved to {MERGED_OUTPUT}"
    )


if __name__ == "__main__":

    run()
