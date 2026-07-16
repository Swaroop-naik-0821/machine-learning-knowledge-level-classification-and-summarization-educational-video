# machine-learning-knowledge-level-classification-and-summarization-educational-video
Machine Learning framework for educational video analysis using Whisper, BERT, and LED to classify knowledge levels and generate summaries.
# Machine Learning Based Knowledge Level Classification and Summarization of Educational Videos

> **An AI-powered framework for automatically classifying educational video content into knowledge levels and generating concise summaries using Whisper, BERT, and LED.**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-green)
![BERT](https://img.shields.io/badge/BERT-Knowledge%20Classification-orange)
![LED](https://img.shields.io/badge/LED-Summarization-blueviolet)

---

## Overview

Educational videos contain valuable learning content, but extracting important concepts from lengthy lectures is often time-consuming.

This project presents a **Machine Learning based framework** that automatically converts educational videos into structured learning resources by integrating Automatic Speech Recognition (ASR), Natural Language Processing (NLP), and Transformer-based Deep Learning models.

The framework:

- Converts educational videos into timestamped transcripts
- Classifies each sentence into cognitive knowledge levels
- Generates concise summaries
- Maps knowledge categories to video timestamps
- Produces structured learning analytics for educators and learners

---

## Key Features

### 🎙️ Speech-to-Text Conversion

- OpenAI Whisper
- Accurate transcript generation
- Timestamp extraction

---

### 🧠 Knowledge-Level Classification

Fine-tuned **BERT** model classifies educational content into four categories:

- Factual Knowledge
- Conceptual Knowledge
- Procedural Knowledge
- Metacognitive Knowledge

---

### 📝 Automatic Summarization

Uses **LED (Longformer Encoder Decoder)** to generate concise summaries from lengthy educational transcripts.

---

### ⏱ Timestamp-Based Learning

Each classified sentence is linked with its original video timestamp, enabling learners to quickly revisit important concepts.

---

### 📊 Educational Analytics

The system generates:

- Knowledge distribution
- Categorized transcript
- Timestamp mapping
- Educational summary

---

# System Architecture

> *(Insert architecture diagram here)*

```
images/architecture.png
```

---

# Workflow

```
Educational Video (.mp4)
        │
        ▼
Audio Extraction
        │
        ▼
OpenAI Whisper
Speech-to-Text
        │
        ▼
Timestamped Transcript
        │
        ▼
Fine-tuned BERT
Knowledge Classification
        │
        ▼
LED Summarizer
        │
        ▼
Summary + Timestamped Knowledge Levels
```

---

# Technologies Used

| Category | Technology |
|-----------|------------|
| Programming | Python |
| Deep Learning | PyTorch |
| NLP | Hugging Face Transformers |
| Speech Recognition | OpenAI Whisper |
| Knowledge Classification | Fine-tuned BERT |
| Summarization | LED (Longformer Encoder Decoder) |
| Development | Google Colab |

---

# Project Structure

```
machine-learning-knowledge-level-classification-and-summarization-educational-video/

├── dataset/
├── docs/
├── images/
├── models/
├── notebooks/
├── outputs/
├── paper/
├── src/
├── README.md
├── LICENSE
├── requirements.txt
└── CITATION.cff
```

---

# Dataset

The model was trained on an annotated educational dataset containing knowledge-labelled statements classified into:

- Factual
- Conceptual
- Procedural
- Metacognitive

A sample dataset is included for demonstration purposes.

---

# Methodology

1. Upload educational video
2. Extract audio
3. Generate transcript using Whisper
4. Preprocess transcript
5. Classify knowledge level using BERT
6. Generate summary using LED
7. Produce timestamp-based learning insights

---

# Sample Output

The framework produces:

- Timestamped transcript
- Knowledge-labelled sentences
- Educational summary
- Knowledge distribution statistics

Example:

| Timestamp | Sentence | Knowledge |
|------------|----------|-----------|
|00:00–00:18|Definition of AI|Factual|
|00:19–01:05|Working principle|Conceptual|
|01:06–02:40|Implementation steps|Procedural|

---

# Research Paper

The complete research paper is available in the **paper/** directory.

---

# Future Work

- Multilingual educational videos
- Real-time classroom transcription
- Interactive learning dashboard
- Question generation
- Bloom's Taxonomy integration

---

# Citation

If this work contributes to your research, please cite the associated research paper.

---

# Author

**Swaroop Prakash Naik**

Dept. of Electronics & Communication Engineering

KLE Technological University, Hubballi

GitHub: https://github.com/Swaroop-naik-0821

Gmail: swaroopnaik2108@gmail.com


**Shashank P Mane**

Dept. of Electronics & Communication Engineering

KLE Technological University, Hubballi

Gmail: shashankmane2004@gmail.com


**Nandana S Korigoudar**

Dept. of Electronics & Communication Engineering

KLE Technological University, Hubballi

Gmail: nandusk29@gmail.com


**Kaushik Mallibhat**

Dept. of Electronics & Communication Engineering

KLE Technological University, Hubballi

Gmail: kaushik@kletech.ac.in



---

# Copyright

**© 2026 Swaroop Prakash Naik, Shashank P Mane, Nandana S Korigoudar, Kaushik Mallibhat**
**All Rights Reserved**

This repository is provided for educational and research reference only.

No part of this repository, including the source code, documentation, figures, datasets, models, or research paper, may be copied, modified, redistributed, or used in another academic, commercial, or personal project without prior written permission from the author.

If you reference this work in academic research, please cite the associated research paper.
