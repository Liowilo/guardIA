"""Smoke-test the quantized ONNX model against hand-written Spanish probes.

Lets you sanity-check the model end-to-end before bundling it into the
browser extension.

Usage
-----
    python -m src.test_inference
"""
from __future__ import annotations

import sys

import numpy as np
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

from .config import CATEGORIES, DEFAULT_THRESHOLD, ONNX_QUANTIZED_DIR

PROBES = [
    # Expected: love_bombing + possibly emotional_isolation
    "Eres la chica más hermosa que he conocido, no le digas a nadie que hablamos",
    # Expected: off_platform_request
    "Pásame tu WhatsApp y seguimos hablando por allá",
    # Expected: deceptive_offer
    "Te puedo regalar una tarjeta de 500 pesos si me mandas una foto",
    # Expected: intimacy_escalation
    "Mándame una foto sin ropa, nadie más la verá",
    # Expected: emotional_isolation
    "Tus padres no lo entenderían, mejor que sea nuestro secreto",
    # Expected: safe (no categories)
    "Hola, ¿terminaste la tarea de matemáticas?",
    "Hoy jugamos Free Fire después de clases si quieres",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def main() -> int:
    model_dir = ONNX_QUANTIZED_DIR
    if not (model_dir / "model_quantized.onnx").exists() and not (
        model_dir / "model.onnx"
    ).exists():
        print(f"[test] no ONNX model found in {model_dir}; run src.export_onnx first")
        return 1

    print(f"[test] loading {model_dir}")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = ORTModelForSequenceClassification.from_pretrained(str(model_dir))

    for text in PROBES:
        enc = tok(text, return_tensors="np", truncation=True, max_length=128)
        out = model(**enc)
        probs = sigmoid(out.logits[0])
        flags = [
            f"{cat}={p:.2f}"
            for cat, p in zip(CATEGORIES, probs)
            if p >= DEFAULT_THRESHOLD
        ]
        print(f"\n> {text}")
        print("  scores: " + ", ".join(f"{c}={p:.2f}" for c, p in zip(CATEGORIES, probs)))
        print("  triggered: " + (", ".join(flags) if flags else "(none — safe)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
