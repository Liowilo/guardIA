"""Export the fine-tuned model to ONNX and apply int8 dynamic quantization.

Produces two artifacts:
- ONNX_DIR: fp32 ONNX (for reference / debugging)
- ONNX_QUANTIZED_DIR: int8 quantized (ship this in the extension)

The quantized model is ~4x smaller and ~2-3x faster on CPU with minimal
accuracy loss for text classification.

Usage
-----
    python -m src.export_onnx
"""
from __future__ import annotations

import shutil
import sys

from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

from .config import MODEL_DIR, ONNX_DIR, ONNX_QUANTIZED_DIR


def main() -> int:
    final_dir = MODEL_DIR / "final"
    if not final_dir.exists():
        print(f"[export] missing {final_dir}; run src.train first")
        return 1

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    ONNX_QUANTIZED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[export] exporting fp32 ONNX from {final_dir}")
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        str(final_dir), export=True
    )
    tok = AutoTokenizer.from_pretrained(str(final_dir))
    ort_model.save_pretrained(str(ONNX_DIR))
    tok.save_pretrained(str(ONNX_DIR))

    print(f"[export] quantizing to int8 → {ONNX_QUANTIZED_DIR}")
    quantizer = ORTQuantizer.from_pretrained(str(ONNX_DIR))
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=str(ONNX_QUANTIZED_DIR), quantization_config=qconfig)

    # Carry over tokenizer + label config to the quantized dir so consumers only
    # need that one directory.
    tok.save_pretrained(str(ONNX_QUANTIZED_DIR))
    for fname in ("config.json",):
        src = ONNX_DIR / fname
        if src.exists():
            shutil.copy(src, ONNX_QUANTIZED_DIR / fname)

    print("[export] done")
    print(f"  fp32 artifacts: {ONNX_DIR}")
    print(f"  int8 artifacts: {ONNX_QUANTIZED_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
