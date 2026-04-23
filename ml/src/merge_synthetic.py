"""Merge PAN-translated labeled data with hand-crafted synthetic examples.

Synthetic rows already have clean multi-label annotations, so they skip the
weak_label step entirely. The output overwrites pan_labeled.parquet so that
train.py picks it up unchanged.

Usage
-----
    python -m src.merge_synthetic
"""
from __future__ import annotations

import sys

import pandas as pd

from .config import CATEGORIES, PAN_LABELED_PATH
from .synthetic_data import EXAMPLES


def synthetic_to_df() -> pd.DataFrame:
    rows = []
    for idx, (text, labels) in enumerate(EXAMPLES):
        row = {
            "conv_id": f"synthetic_{idx}",
            "line_num": "0",
            "author": "synthetic",
            "text": text,
            "text_es": text,
            "is_grooming": bool(labels),
            "author_is_predator": bool(labels),
        }
        for cat in CATEGORIES:
            row[cat] = int(cat in labels)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    if not PAN_LABELED_PATH.exists():
        print(f"[merge] missing {PAN_LABELED_PATH}; run src.weak_label first")
        return 1

    pan = pd.read_parquet(PAN_LABELED_PATH)
    syn = synthetic_to_df()

    # Align columns (PAN has extras; keep only the columns we both need).
    common = ["conv_id", "line_num", "author", "text", "text_es", *CATEGORIES]
    merged = pd.concat([pan[common], syn[common]], ignore_index=True)
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"[merge] PAN rows: {len(pan):,}")
    print(f"[merge] synthetic rows: {len(syn):,}")
    print(f"[merge] merged: {len(merged):,}")
    print("[merge] per-category positive totals in merged set:")
    for cat in CATEGORIES:
        pan_pos = int(pan[cat].sum())
        syn_pos = int(syn[cat].sum())
        total = int(merged[cat].sum())
        print(f"  {cat}: {total:,}  (PAN={pan_pos:,} + synth={syn_pos:,})")

    merged.to_parquet(PAN_LABELED_PATH, index=False)
    print(f"[merge] wrote {PAN_LABELED_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
