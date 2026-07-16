"""
Main Pipeline

Machine Learning Based Knowledge Level Classification
and Summarization of Educational Videos
"""

from audio_extraction import transcribe_video
from sentence_segmentation import create_sentence_timestamps
from classify_transcript import classify_transcript
from analytics import generate_statistics
from merge_segments import run as merge_segments
from summarizer import run as generate_summary


def main():

    print("=" * 70)
    print("Knowledge Level Classification Pipeline")
    print("=" * 70)

    print("\nStep 1 : Whisper Transcription")
    transcribe_video()

    print("\nStep 2 : Sentence Segmentation")
    create_sentence_timestamps()

    print("\nStep 3 : Knowledge Classification")
    classify_transcript()

    print("\nStep 4 : Analytics")
    generate_statistics()

    print("\nStep 5 : Merge Segments")
    merge_segments()

    print("\nStep 6 : LED Summarization")
    generate_summary()

    print("\nPipeline Completed Successfully!")


if __name__ == "__main__":
    main()
