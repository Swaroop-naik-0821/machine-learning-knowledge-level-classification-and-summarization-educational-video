"""
Project Configuration File

Machine Learning Based Knowledge Level Classification
and Summarization of Educational Videos

Author:
Swaroop Prakash Naik
"""

# ==========================================================
# INPUT PATHS
# ==========================================================

# Input educational video
VIDEO_PATH = "input/video.mp4"

# Training dataset
TRAIN_DATASET = "dataset/sample_dataset.csv"


# ==========================================================
# TRANSCRIPT FILES
# ==========================================================

# Whisper output
RAW_TRANSCRIPT_JSON = "outputs/clean_whisper.json"

# Sentence-level transcript
SENTENCE_TRANSCRIPT_JSON = "outputs/sentence_timestamps.json"


# ==========================================================
# MODEL PATHS
# ==========================================================

# Fine-tuned BERT model directory
MODEL_DIRECTORY = "models/bert_model"

# Whisper model
WHISPER_MODEL = "base"

# LED model
LED_MODEL = "allenai/led-base-16384"


# ==========================================================
# OUTPUT FILES
# ==========================================================

# Classified transcript
CLASSIFICATION_OUTPUT = "outputs/classified_output.json"

# Merged transcript
MERGED_OUTPUT = "outputs/merged_output.json"

# Final summary
SUMMARY_OUTPUT = "outputs/summary.txt"

# Statistics
STATISTICS_OUTPUT = "outputs/knowledge_distribution.csv"


# ==========================================================
# IMAGE OUTPUTS
# ==========================================================

BAR_CHART = "images/knowledge_distribution.png"

PIE_CHART = "images/knowledge_distribution_pie.png"


# ==========================================================
# BERT TRAINING PARAMETERS
# ==========================================================

MAX_LENGTH = 128

BATCH_SIZE = 8

LEARNING_RATE = 2e-5

EPOCHS = 3


# ==========================================================
# LED SUMMARIZATION PARAMETERS
# ==========================================================

MAX_INPUT_LENGTH = 4096

MAX_SUMMARY_LENGTH = 200

MIN_SUMMARY_LENGTH = 80

NUM_BEAMS = 4


# ==========================================================
# KNOWLEDGE LABELS
# ==========================================================

LABEL_MAP = {
    "Factual": 0,
    "Conceptual": 1,
    "Procedural": 2,
    "Metacognitive": 3
}

ID2LABEL = {
    0: "Factual",
    1: "Conceptual",
    2: "Procedural",
    3: "Metacognitive"
}
