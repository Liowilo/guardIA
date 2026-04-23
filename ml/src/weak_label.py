"""Assign the 5 grooming categories to translated messages using Spanish regex rules.

Strategy
--------
PAN only provides binary grooming labels. We need multi-label data across 5
categories. We apply hand-crafted Spanish patterns to each message:

- Grooming messages (is_grooming=True) that match >=1 category get those
  categories as positive labels.
- Grooming messages matching 0 categories are dropped (noisy for our schema).
- Non-grooming messages become all-zeros across the 5 categories.

This is intentionally imperfect — weak supervision. The model learns to
generalize beyond keyword matching.

Usage
-----
    python -m src.weak_label
"""
from __future__ import annotations

import re
import sys

import pandas as pd

from .config import CATEGORIES, PAN_LABELED_PATH, PAN_TRANSLATED_PATH

# Regex are applied case-insensitively. Patterns are tuned against MarianMT
# Spanish translations of PAN-2012 predator chat — they include both natural
# Spanish and common MT artefacts.
PATTERNS: dict[str, list[str]] = {
    "love_bombing": [
        r"\bhermos[ao]s?\b",
        r"\bbell[ao]s?\b",
        r"\bpreciosa?\b",
        r"\beres (muy |tan )?especial\b",
        r"\beres únic[ao]\b",
        r"\bmi amor\b",
        r"\bcariñ[oa]\b",
        r"\bguapa?\b",
        r"\bsexy\b",
        r"\blind[ao]s?\b",
        r"\bte quiero\b",
        r"\bte amo\b",
        r"\bte adoro\b",
        r"\bmuñeca\b",
        r"\bprincesa\b",
        r"\bdulce (jovencita|niña|chica)\b",
        r"\bdulzura\b",
        r"\bnunca he conocido\b",
        r"\beres (la|un)a? (mejor|chica|persona)\b",
        r"\beres perfect[ao]\b",
        r"\btienes algo especial\b",
        r"\bme gustas\b",
        r"\bme vuelves loc[ao]\b",
        r"\bte extraño\b",
        r"\bpienso en ti\b",
        r"\bpensando en ti\b",
        r"\bte ves (bien|hermosa?|genial|lind[ao]|preciosa)\b",
        r"\bsabes que eres\b",
        r"\bcreo que eres\b",
        r"\beres (tan |muy )?(inteligente|madura?|lind[ao])\b",
        r"\bsolo si tú también me quieres\b",
        r"\bquiero que me quieras\b",
        r"\bpor tu amor\b",
    ],
    "intimacy_escalation": [
        r"\bcuerpo\b",
        r"\bdesnud[ao]s?\b",
        r"\bauto ?desnud\w*\b",
        r"\bfotos? (íntimas?|privadas?|picantes?|sexy|sin ropa)\b",
        r"\btocarte?\b",
        r"\bbesarte?\b",
        r"\b(te has|has) besado\b",
        r"\bbesos?\b",
        r"\bsexo\b",
        r"\bsexual\b",
        r"\bsexualmente\b",
        r"\bcama\b",
        r"\benséñame\b",
        r"\bmuéstrame\b",
        r"\bmándame una foto\b",
        r"\bmanda(me)? (una )?foto\b",
        r"\btus? fotos?\b",
        r"\bfoto(s)? tuya(s)?\b",
        r"\bsin ropa\b",
        r"\btendrás la ropa\b",
        r"\bquitarte\b",
        r"\bquítate\b",
        r"\bintim[ao]s?\b",
        r"\bpechos?\b",
        r"\bsenos?\b",
        r"\btrasero\b",
        r"\bnalgas?\b",
        r"\bexcit[ao]\b",
        r"\bcalient[ae]\b",
        r"\bvirgen\b",
        r"\bmasturb\w*\b",
        r"\bputa\b",
        r"\bapretad[ao]\b",
        r"\bhacer el amor\b",
        r"\bhacerte\b",
        r"\btienes novio\b",
        r"\btienes experiencia\b",
    ],
    "emotional_isolation": [
        # Checking who else is around — the single strongest signal of grooming
        r"\b(cuándo|a qué hora) (vuelve|llega|regresa|viene|sale)\b",
        r"\bestá tu (mamá|papá|madre|padre|herman[oa]|tutor[ae])\b",
        r"\btu (mamá|papá|madre|padre) (está|llega|vuelve|sabe|se entera)\b",
        r"\bestás sol[ao]\b",
        r"\b(hay|está) alguien (en casa|contigo|ahí|cerca)\b",
        r"\bquién (más )?(está|hay) (en casa|contigo|ahí|cerca)\b",
        r"\b(mamá|papá) (se fue|salió|duerme|está dormid[ao])\b",
        r"\b(tus) padres (duermen|están dormidos|salieron|se fueron)\b",
        # Secret-keeping
        r"\b(nuestro|un) secreto\b",
        r"\bes un secreto\b",
        r"\bno le digas\b",
        r"\bno les digas\b",
        r"\bno se lo digas\b",
        r"\bno se lo cuentes\b",
        r"\bentre (tú y yo|nosotros)\b",
        r"\bmamás la palabra\b",
        r"\bno te lo voy a decir\b",
        r"\bno lo entenderían\b",
        r"\bnadie (más )?(lo |se )?(sabe|sabrá|debe saber|se entera)\b",
        r"\bmejor no digas\b",
        r"\bsolo confío en ti\b",
        r"\btus padres no\b",
        r"\btus amig[oa]s no\b",
        r"\bno te creerán\b",
        r"\bnadie te entiende como yo\b",
        r"\bno le digas a nadie\b",
        r"\bquedará entre\b",
    ],
    "deceptive_offer": [
        r"\bregalo\b",
        r"\bregalar(te|le)?\b",
        r"\bdinero\b",
        r"\bte pago\b",
        r"\bte compro\b",
        r"\bte voy a comprar\b",
        r"\bquieres que te compre\b",
        r"\bmodel(aje|ar|o)\b",
        r"\boportunidad\b",
        r"\bbeca\b",
        r"\btrabajo\b",
        r"\btarjeta (regalo|de regalo)\b",
        r"\brecarga\b",
        r"\bcelular (nuevo)?\b",
        r"\bteléfono nuevo\b",
        r"\bgift ?card\b",
        r"\bvoucher\b",
        r"\bpagar(te|le)?\b",
        r"\bte invit[oa]\b",
        r"\btomar vuelo\b",
        r"\bconcurso\b",
        r"\bpremi[oa]\b",
        r"\bayudarte económic\w*\b",
        r"\bte llevo a\b",
    ],
    "off_platform_request": [
        r"\bwhatsapp\b",
        r"\btelegram\b",
        r"\bsnapchat\b",
        r"\bdiscord\b",
        r"\bskype\b",
        r"\bkik\b",
        r"\byahoo (messenger|messeng)?\b",
        r"\bmsn\b",
        r"\bicq\b",
        r"\btu número\b",
        r"\bmi número\b",
        r"\bnúmero de (celular|teléfono|cel)\b",
        r"\bmándame tu número\b",
        r"\bllámame\b",
        r"\bllama(me)? (al|a)\b",
        r"\bvideollamada\b",
        r"\bvideo llamada\b",
        r"\bwebcam\b",
        r"\bcámara\b",
        r"\bhablemos por\b",
        r"\bfuera de aquí\b",
        r"\bmovamos (la conversación|a)\b",
        r"\bquedamos\b",
        r"\bnos vemos\b",
        r"\bvernos\b",
        r"\bencontrarnos\b",
        r"\btu dirección\b",
        r"\bdónde vives\b",
        r"\b¿?a qué (escuela|colegio|secundaria|prepa)\b",
        r"\bpasamelo\b",
        r"\bpásame tu\b",
        r"\bvernos en persona\b",
        r"\bconocernos en persona\b",
        r"\bnos podemos ver\b",
        r"\bpuedes b conmigo\b",
        r"\btext (me|ing)\b",
        r"\bmensaje de texto\b",
    ],
}

COMPILED = {
    cat: re.compile("|".join(pats), flags=re.IGNORECASE | re.UNICODE)
    for cat, pats in PATTERNS.items()
}


def label_row(text: str) -> dict[str, int]:
    return {cat: int(bool(rx.search(text))) for cat, rx in COMPILED.items()}


def main() -> int:
    if not PAN_TRANSLATED_PATH.exists():
        print(f"[weak_label] missing {PAN_TRANSLATED_PATH}; run src.translate first")
        return 1

    df = pd.read_parquet(PAN_TRANSLATED_PATH)
    label_df = pd.DataFrame(
        [label_row(t) for t in df["text_es"].astype(str)]
    )
    df = pd.concat([df.reset_index(drop=True), label_df], axis=1)

    any_cat = df[CATEGORIES].sum(axis=1) > 0

    # Positive signal = author is a confirmed predator. Keep predator lines that
    # match >=1 category (supervised positives) and all non-predator lines as
    # negatives (all-zero labels).
    keep = (df["author_is_predator"] & any_cat) | (~df["author_is_predator"])
    dropped = (~keep).sum()
    df = df[keep].reset_index(drop=True)

    print(f"[weak_label] dropped {dropped:,} predator rows with no category match")
    print(f"[weak_label] kept {len(df):,} rows")
    for cat in CATEGORIES:
        print(f"  {cat}: positives={int(df[cat].sum()):,}")
    df.to_parquet(PAN_LABELED_PATH, index=False)
    print(f"[weak_label] wrote {PAN_LABELED_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
