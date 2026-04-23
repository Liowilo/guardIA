# guardIA — Fine-tuning pipeline

Multi-label Spanish grooming-pattern detector built on top of multilingual
MiniLM, fine-tuned on a translated subset of the PAN-2012 Sexual Predator
Identification corpus.

## Output

A quantized ONNX model (~30 MB) that classifies Spanish chat messages across
5 labels (sigmoid per label — a single message can carry several):

- `love_bombing`
- `intimacy_escalation`
- `emotional_isolation`
- `deceptive_offer`
- `off_platform_request`

Ready to load in the browser extension via `@huggingface/transformers`.

## Directory layout

```
ml/
├── requirements.txt
├── src/
│   ├── config.py            # paths + hyperparameters (env-var overridable)
│   ├── download_pan.py      # try public mirrors; fall back to manual upload
│   ├── parse_pan.py         # XML → parquet + balanced sample
│   ├── translate.py         # MarianMT en→es, batched, resumable
│   ├── weak_label.py        # Spanish regex → 5 categories
│   ├── train.py             # MiniLM multi-label fine-tune
│   ├── export_onnx.py       # ONNX + int8 dynamic quantization
│   └── test_inference.py    # smoke tests with Spanish probes
└── data/                    # all artifacts land here (gitignore this)
```

## Colab runbook

Mount Drive so artifacts survive Colab timeouts. Every step after `parse_pan`
is resumable.

```python
# --- Cell 1: setup ---
from google.colab import drive
drive.mount("/content/drive")

import os
os.environ["GUARDIA_DATA_DIR"] = "/content/drive/MyDrive/guardIA/data"

!git clone <your-repo-url> /content/guardIA        # or upload ml/ manually
%cd /content/guardIA/ml
!pip install -q -r requirements.txt

# --- Cell 2: get PAN corpus ---
!python -m src.download_pan
# If this fails: register at https://pan.webis.de, download the SPI training
# corpus, and upload these three files into GUARDIA_DATA_DIR/pan_raw/:
#   - pan12-sexual-predator-identification-training-corpus-2012-05-01.xml
#   - pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt
#   - pan12-sexual-predator-identification-diff-2012-05-01.txt

# --- Cell 3: parse + sample (~2 min, CPU) ---
!python -m src.parse_pan

# --- Cell 4: translate en→es (~15-30 min on T4) ---
!python -m src.translate

# --- Cell 5: weak-label the 5 categories (~30 sec, CPU) ---
!python -m src.weak_label

# --- Cell 6: fine-tune MiniLM (~30-60 min on T4) ---
!python -m src.train

# --- Cell 7: export + quantize ONNX (~2 min) ---
!python -m src.export_onnx

# --- Cell 8: smoke test ---
!python -m src.test_inference
```

Total wall time on free Colab T4: **~1-2 hours end-to-end**, dominated by
translation and training.

## Resume semantics

- `translate.py` writes to `pan_translated.parquet` every 10 batches. If Colab
  disconnects, re-run the cell; it skips already-translated IDs.
- `train.py` uses `load_best_model_at_end` — the best checkpoint (by micro-F1
  on the validation split) is saved to `data/model/final/`.
- `export_onnx.py` is idempotent; re-runs overwrite.

## Tuning knobs (env vars)

Set before `!python -m src.train`:

| Var | Default | Notes |
|---|---|---|
| `GUARDIA_BASE_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Switch to `distilbert-base-multilingual-cased` if you insist on DistilBERT |
| `GUARDIA_EPOCHS` | `4` | 3 is usually enough; 5 starts overfitting weak labels |
| `GUARDIA_TRAIN_BATCH` | `32` | Bump to 64 on T4 if max_len=128 |
| `GUARDIA_LR` | `3e-5` | Sentence-transformer MiniLM tolerates 5e-5 too |
| `GUARDIA_MAX_LEN` | `128` | Chat messages are short; 128 covers >99% |
| `GUARDIA_NEG_PER_POS` | `1` | 2-3 gives more realistic class balance but slows translation |

## Known limitations (say these in the pitch, don't hide them)

1. **Weak supervision.** Categories come from regex on translated text, not
   human annotators. The model learns to generalize but inherits regex bias.
2. **Domain shift.** PAN-2012 is IRC chat logs from 2001-2010. WhatsApp and
   Discord messages in 2026 read differently. Roadmap: fine-tune on
   anonymized pilot data.
3. **Machine-translated Spanish.** MarianMT en→es produces natural-ish Spanish
   but not Mexican idioms. A Mexican-Spanish adaptation pass (rule-based
   rewrites of common artefacts) would help.
4. **Multilingual MiniLM on chat.** Base is trained on paraphrase/NLI, not
   adversarial chat. Works well enough here because grooming signals are
   lexical-semantic, not discourse-level.

## Loading in the extension

After `export_onnx.py`, copy `data/onnx_int8/` into the extension bundle
(e.g. `extension/public/model/`). In content scripts:

```js
import { pipeline, env } from "@huggingface/transformers";

env.allowRemoteModels = false;
env.localModelPath = "/model/";

const classifier = await pipeline(
  "text-classification",
  ".",  // relative to localModelPath
  { quantized: true, topk: null }  // topk:null → return all 5 scores
);

const scores = await classifier("hola, pásame tu whatsapp");
// [{label:"love_bombing", score:0.02}, {label:"off_platform_request", score:0.87}, ...]
```
