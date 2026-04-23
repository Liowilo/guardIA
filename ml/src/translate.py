"""Translate the sampled PAN corpus from English to Spanish with MarianMT.

Uses Helsinki-NLP/opus-mt-en-es on GPU. Resumable: if PAN_TRANSLATED_PATH
already contains rows, skips those IDs.

Usage
-----
    python -m src.translate
"""
from __future__ import annotations

import sys
import time

import pandas as pd
import torch
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

from .config import (
    MT_BATCH_SIZE,
    MT_MAX_LEN,
    MT_MODEL,
    PAN_SAMPLED_PATH,
    PAN_TRANSLATED_PATH,
)


def load_already_done() -> set[tuple[str, str]]:
    if PAN_TRANSLATED_PATH.exists():
        prev = pd.read_parquet(PAN_TRANSLATED_PATH, columns=["conv_id", "line_num"])
        return set(map(tuple, prev.to_numpy()))
    return set()


def main() -> int:
    if not PAN_SAMPLED_PATH.exists():
        print(f"[translate] missing {PAN_SAMPLED_PATH}; run src.parse_pan first")
        return 1

    df = pd.read_parquet(PAN_SAMPLED_PATH)
    done = load_already_done()
    if done:
        mask = ~df.set_index(["conv_id", "line_num"]).index.isin(done)
        df = df[mask].reset_index(drop=True)
        print(f"[translate] resuming, {len(df):,} rows remain")
    if df.empty:
        print("[translate] nothing to do")
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[translate] loading {MT_MODEL} on {device}")
    tok = MarianTokenizer.from_pretrained(MT_MODEL)
    model = MarianMTModel.from_pretrained(MT_MODEL).to(device)
    if device == "cuda":
        model = model.half()
    model.eval()

    texts = df["text"].tolist()
    out_chunks: list[pd.DataFrame] = []
    start = time.time()

    with torch.inference_mode():
        for i in tqdm(range(0, len(texts), MT_BATCH_SIZE), unit="batch"):
            batch_texts = texts[i : i + MT_BATCH_SIZE]
            batch_rows = df.iloc[i : i + MT_BATCH_SIZE]
            enc = tok(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MT_MAX_LEN,
            ).to(device)
            out = model.generate(
                **enc,
                max_new_tokens=MT_MAX_LEN,
                num_beams=1,  # greedy; fast and adequate for chat
            )
            spanish = tok.batch_decode(out, skip_special_tokens=True)
            chunk = batch_rows.copy()
            chunk["text_es"] = spanish
            out_chunks.append(chunk)

            # Flush every ~10 batches so crashes don't lose everything.
            if len(out_chunks) >= 10:
                _append(out_chunks)
                out_chunks = []

    if out_chunks:
        _append(out_chunks)

    elapsed = time.time() - start
    print(f"[translate] done in {elapsed/60:.1f} min → {PAN_TRANSLATED_PATH}")
    return 0


def _append(chunks: list[pd.DataFrame]) -> None:
    new = pd.concat(chunks, ignore_index=True)
    if PAN_TRANSLATED_PATH.exists():
        prev = pd.read_parquet(PAN_TRANSLATED_PATH)
        new = pd.concat([prev, new], ignore_index=True)
    new.to_parquet(PAN_TRANSLATED_PATH, index=False)


if __name__ == "__main__":
    sys.exit(main())
