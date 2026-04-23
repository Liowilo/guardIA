"""Parse the PAN-2012 SPI XML + label files into a flat parquet of messages.

Outputs
-------
PAN_PARSED_PATH : all messages with is_grooming, author_is_predator flags.
PAN_SAMPLED_PATH : balanced subset (all grooming + NEG_PER_POS non-grooming per
                   grooming) ready to translate.

Usage
-----
    python -m src.parse_pan
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd
from lxml import etree
from tqdm import tqdm

from .config import (
    MIN_TOKENS,
    NEG_PER_POS,
    PAN_PARSED_PATH,
    PAN_RAW_DIR,
    PAN_SAMPLED_PATH,
    PAN_TRAIN_DIFF,
    PAN_TRAIN_PREDATORS,
    PAN_TRAIN_XML,
    SEED,
)


def load_predator_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def load_grooming_lines(path: Path) -> set[tuple[str, str]]:
    """diff.txt format per line: `<conv_id> <line_num>`."""
    pairs: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pairs.add((parts[0], parts[1]))
    return pairs


def iter_messages(xml_path: Path):
    """Stream <message> elements without holding the whole tree in memory."""
    context = etree.iterparse(str(xml_path), events=("end",), tag="conversation")
    for _, conv in context:
        conv_id = conv.get("id", "")
        for msg in conv.findall("message"):
            line_num = msg.get("line", "")
            author_el = msg.find("author")
            text_el = msg.find("text")
            author = (author_el.text or "").strip() if author_el is not None else ""
            text = (text_el.text or "").strip() if text_el is not None else ""
            if text:
                yield conv_id, line_num, author, text
        conv.clear()
        while conv.getprevious() is not None:
            del conv.getparent()[0]


def main() -> int:
    xml_path = PAN_RAW_DIR / PAN_TRAIN_XML
    predators_path = PAN_RAW_DIR / PAN_TRAIN_PREDATORS
    diff_path = PAN_RAW_DIR / PAN_TRAIN_DIFF

    for p in (xml_path, predators_path, diff_path):
        if not p.exists():
            print(f"[parse_pan] missing file: {p}")
            print("Run src.download_pan or upload PAN files manually.")
            return 1

    print("[parse_pan] loading label files...")
    predator_ids = load_predator_ids(predators_path)
    grooming_lines = load_grooming_lines(diff_path)
    print(f"[parse_pan] {len(predator_ids)} predator IDs, {len(grooming_lines)} grooming lines")

    print("[parse_pan] streaming XML...")
    rows = []
    for conv_id, line_num, author, text in tqdm(iter_messages(xml_path), unit="msg"):
        if len(text.split()) < MIN_TOKENS:
            continue
        is_grooming = (conv_id, line_num) in grooming_lines
        author_is_predator = author in predator_ids
        rows.append(
            {
                "conv_id": conv_id,
                "line_num": line_num,
                "author": author,
                "text": text,
                "is_grooming": is_grooming,
                "author_is_predator": author_is_predator,
            }
        )

    df = pd.DataFrame(rows)
    df.to_parquet(PAN_PARSED_PATH, index=False)
    print(f"[parse_pan] wrote {len(df):,} messages to {PAN_PARSED_PATH}")
    print(
        f"[parse_pan] grooming={df.is_grooming.sum():,}  "
        f"predator_authored={df.author_is_predator.sum():,}"
    )

    # Balanced sample for translation.
    # Positive signal: author is a confirmed predator (predators.txt). This is
    # the reliable ground-truth from PAN-2012 SPI. diff.txt turned out to be
    # misaligned with the training XML, so we don't use is_grooming as the
    # positive label here.
    pos = df[df.author_is_predator]
    neg = df[~df.author_is_predator]
    n_neg = min(len(neg), len(pos) * NEG_PER_POS)
    neg_sample = neg.sample(n=n_neg, random_state=SEED)
    sampled = (
        pd.concat([pos, neg_sample], ignore_index=True)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )
    sampled.to_parquet(PAN_SAMPLED_PATH, index=False)
    print(
        f"[parse_pan] sampled predator_pos={len(pos):,} safe_neg={n_neg:,} → {PAN_SAMPLED_PATH}"
    )
    return 0


if __name__ == "__main__":
    random.seed(SEED)
    sys.exit(main())
