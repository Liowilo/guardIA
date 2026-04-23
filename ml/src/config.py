"""Central configuration for the guardIA fine-tuning pipeline.

All paths are resolved relative to GUARDIA_DATA_DIR (env var) or ./data.
Tune hyperparameters via env vars when running in Colab.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("GUARDIA_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
ROOT.mkdir(parents=True, exist_ok=True)

# --- Input / intermediate / output paths ---
PAN_RAW_DIR = ROOT / "pan_raw"               # Unzipped PAN-2012 files go here
PAN_PARSED_PATH = ROOT / "pan_parsed.parquet"  # All messages w/ is_grooming flag
PAN_SAMPLED_PATH = ROOT / "pan_sampled.parquet"  # Balanced subset to translate
PAN_TRANSLATED_PATH = ROOT / "pan_translated.parquet"  # + text_es column
PAN_LABELED_PATH = ROOT / "pan_labeled.parquet"  # + 5 category columns
MODEL_DIR = ROOT / "model"                    # Trainer checkpoints
ONNX_DIR = ROOT / "onnx"                      # Exported / quantized ONNX
ONNX_QUANTIZED_DIR = ROOT / "onnx_int8"

# --- PAN file names (standard) ---
PAN_TRAIN_XML = "pan12-sexual-predator-identification-training-corpus-2012-05-01.xml"
PAN_TRAIN_PREDATORS = "pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt"
PAN_TRAIN_DIFF = "pan12-sexual-predator-identification-diff.txt"

# --- Category schema ---
CATEGORIES = [
    "love_bombing",
    "intimacy_escalation",
    "emotional_isolation",
    "deceptive_offer",
    "off_platform_request",
]

# --- Sampling ---
# How many non-grooming messages to keep per grooming message. Keeps class balance
# reasonable without translating ~900K non-grooming lines.
NEG_PER_POS = int(os.environ.get("GUARDIA_NEG_PER_POS", "1"))
MIN_TOKENS = int(os.environ.get("GUARDIA_MIN_TOKENS", "3"))  # Drop trivially-short lines
SEED = int(os.environ.get("GUARDIA_SEED", "42"))

# --- Translation ---
MT_MODEL = os.environ.get("GUARDIA_MT_MODEL", "Helsinki-NLP/opus-mt-en-es")
MT_BATCH_SIZE = int(os.environ.get("GUARDIA_MT_BATCH", "64"))
MT_MAX_LEN = int(os.environ.get("GUARDIA_MT_MAXLEN", "128"))

# --- Training ---
BASE_MODEL = os.environ.get(
    "GUARDIA_BASE_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
NUM_EPOCHS = int(os.environ.get("GUARDIA_EPOCHS", "4"))
TRAIN_BATCH = int(os.environ.get("GUARDIA_TRAIN_BATCH", "32"))
EVAL_BATCH = int(os.environ.get("GUARDIA_EVAL_BATCH", "64"))
LR = float(os.environ.get("GUARDIA_LR", "3e-5"))
MAX_LEN = int(os.environ.get("GUARDIA_MAX_LEN", "128"))
WARMUP_RATIO = float(os.environ.get("GUARDIA_WARMUP", "0.1"))
WEIGHT_DECAY = float(os.environ.get("GUARDIA_WD", "0.01"))

# --- Inference / thresholds ---
# Per-category thresholds can be tuned post-training; start at 0.5 sigmoid.
DEFAULT_THRESHOLD = 0.5