from audio_extraction import transcribe_video
from sentence_segmentation import create_sentence_timestamps
from classify_transcript import classify_transcript
from merge_segments import merge_segments
from analytics import generate_statistics
from summarizer import generate_summary

def main():

    transcribe_video()

    create_sentence_timestamps()

    classify_transcript()

    merge_segments()

    generate_statistics()

    generate_summary()

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
