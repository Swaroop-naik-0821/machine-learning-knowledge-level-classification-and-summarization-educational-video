"""
Analytics Module

Generates statistics and visualization for the classified
educational transcript.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt

from config import CLASSIFICATION_OUTPUT


def load_classification():

    with open(CLASSIFICATION_OUTPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    return pd.DataFrame(data)


def knowledge_distribution(df):

    return df["label"].value_counts()


def save_statistics(df):

    stats = knowledge_distribution(df)

    stats.to_csv("outputs/knowledge_distribution.csv")

    print("Knowledge statistics saved.")


def plot_bar_chart(df):

    counts = knowledge_distribution(df)

    plt.figure(figsize=(8,5))

    counts.plot(kind="bar")

    plt.title("Knowledge Level Distribution")

    plt.xlabel("Knowledge Level")

    plt.ylabel("Number of Sentences")

    plt.tight_layout()

    plt.savefig(
        "images/knowledge_distribution.png",
        dpi=300
    )

    plt.close()

    print("Bar chart saved.")


def plot_pie_chart(df):

    counts = knowledge_distribution(df)

    plt.figure(figsize=(6,6))

    plt.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title("Knowledge Distribution")

    plt.tight_layout()

    plt.savefig(
        "images/knowledge_distribution_pie.png",
        dpi=300
    )

    plt.close()

    print("Pie chart saved.")


def transcript_statistics(df):

    print("\n========= Statistics =========")

    print(f"Total Sentences : {len(df)}")

    print()

    print(knowledge_distribution(df))

    print("==============================\n")


def generate_statistics():

    df = load_classification()

    transcript_statistics(df)

    save_statistics(df)

    plot_bar_chart(df)

    plot_pie_chart(df)

    return df


if __name__ == "__main__":

    generate_statistics()
