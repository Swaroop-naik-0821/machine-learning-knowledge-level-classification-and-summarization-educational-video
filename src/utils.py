"""
Utility Functions
"""

import json
import os


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, filepath):
    """Save dictionary as JSON."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def sec_to_timestamp(seconds):
    """Convert seconds into MM:SS format."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def ensure_directory(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
