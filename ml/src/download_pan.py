"""Download and unpack the PAN-2012 Sexual Predator Identification corpus.

PAN-2012 SPI is gated: the official distribution at https://pan.webis.de requires
registration. This script first tries known public mirrors, and if none succeed
prints instructions for manual upload.

Usage (from ml/):
    python -m src.download_pan
or inside Colab:
    !python ml/src/download_pan.py

If download fails, upload the three files listed in config.py to PAN_RAW_DIR
manually (Colab: left-pane file upload, or copy from Drive) and re-run the
pipeline from src.parse_pan.
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

from .config import PAN_RAW_DIR, PAN_TRAIN_XML, PAN_TRAIN_PREDATORS, PAN_TRAIN_DIFF

# Known public mirrors of PAN-2012 SPI. These can rot — keep the list ordered
# most-likely-to-work first and fall back to manual instructions.
MIRRORS = [
    # Zenodo mirror of the PAN-2012 training corpus (if available)
    "https://zenodo.org/record/3713249/files/pan12-sexual-predator-identification-training-corpus-2012-05-01.zip",
]

REQUIRED = [PAN_TRAIN_XML, PAN_TRAIN_PREDATORS, PAN_TRAIN_DIFF]


def already_present() -> bool:
    return all((PAN_RAW_DIR / name).exists() for name in REQUIRED)


def try_download(url: str, dest: Path) -> bool:
    try:
        print(f"[download] GET {url}")
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[download] failed: {e}")
        return False


def main() -> int:
    PAN_RAW_DIR.mkdir(parents=True, exist_ok=True)

    if already_present():
        print(f"[download] PAN files already present in {PAN_RAW_DIR}")
        return 0

    zip_path = PAN_RAW_DIR / "pan12_spi_training.zip"
    for url in MIRRORS:
        if try_download(url, zip_path):
            print(f"[download] unpacking {zip_path}")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(PAN_RAW_DIR)
            zip_path.unlink(missing_ok=True)
            if already_present():
                print("[download] done")
                return 0

    print("\n[download] Automatic download failed. Manual steps:")
    print("  1. Register at https://pan.webis.de/data.html")
    print("  2. Download the PAN-2012 Sexual Predator Identification training corpus")
    print(f"  3. Upload these files to {PAN_RAW_DIR}:")
    for name in REQUIRED:
        print(f"       - {name}")
    print("  4. Re-run from: python -m src.parse_pan")
    return 1


if __name__ == "__main__":
    sys.exit(main())