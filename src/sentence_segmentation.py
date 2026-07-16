"""
Sentence Segmentation Module

Converts Whisper transcription into sentence-level
transcripts with timestamps.

Each sentence is stored along with its start and end time.
"""

import re

from config import (
    RAW_TRANSCRIPT_JSON,
    SENTENCE_TRANSCRIPT_JSON
)

from utils import (
    load_json,
    save_json
)


def split_into_sentences(text):
    """
    Split text into sentences.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def create_sentence_timestamps():

    whisper_data = load_json(RAW_TRANSCRIPT_JSON)

    segments = whisper_data["segments"]

    sentence_data = []

    for segment in segments:

        start = segment["start"]
        end = segment["end"]
        text = segment["text"].strip()

        sentences = split_into_sentences(text)

        if len(sentences) == 1:

            sentence_data.append(
                {
                    "sentence": sentences[0],
                    "start": start,
                    "end": end
                }
            )

        else:

            duration = end - start

            step = duration / len(sentences)

            current = start

            for sentence in sentences:

                sentence_data.append(
                    {
                        "sentence": sentence,
                        "start": round(current, 2),
                        "end": round(current + step, 2)
                    }
                )

                current += step

    save_json(sentence_data, SENTENCE_TRANSCRIPT_JSON)

    print(f"Saved {len(sentence_data)} sentences.")


if __name__ == "__main__":

    create_sentence_timestamps()
