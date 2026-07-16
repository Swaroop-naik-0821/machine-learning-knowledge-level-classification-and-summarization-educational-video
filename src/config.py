"""
Project Configuration
Machine Learning Based Knowledge Level Classification
and Summarization of Educational Videos
"""

# -------------------------------
# Input Files
# -------------------------------

VIDEO_PATH = "input/video.mp4"

# -------------------------------
# Transcript Files
# -------------------------------

RAW_TRANSCRIPT_JSON = "outputs/clean_whisper.json"

SENTENCE_TRANSCRIPT_JSON = "outputs/sentence_timestamps.json"

# -------------------------------
# Dataset
# -------------------------------

TRAIN_DATASET = "dataset/sample_dataset.csv"

# -------------------------------
# Trained Model
# -------------------------------

MODEL_DIRECTORY = "models/bert_model"

# -------------------------------
# Output Files
# -------------------------------

CLASSIFICATION_OUTPUT = "outputs/classified_output.csv"

MERGED_OUTPUT = "outputs/merged_output.csv"

SUMMARY_OUTPUT = "outputs/summary.txt"

STATISTICS_OUTPUT = "outputs/statistics.csv"

# -------------------------------
# Whisper Settings
# -------------------------------

WHISPER_MODEL = "base"

# -------------------------------
# BERT Settings
# -------------------------------

MAX_LENGTH = 128

BATCH_SIZE = 8

LEARNING_RATE = 2e-5

EPOCHS = 3

# -------------------------------
# LED Settings
# -------------------------------

MAX_INPUT_LENGTH = 4096

MAX_SUMMARY_LENGTH = 200
