# Modelo de detección de grooming — guardIA

**Documentación del componente de Machine Learning desarrollado previo al hackatón.**

Este documento existe para cumplir con el reglamento: declarar explícitamente qué trabajo se realizó antes del evento y por qué. Todo lo descrito aquí se construyó como infraestructura base; la implementación del producto (extensión de navegador, dashboard del Pacto Digital, integración con Supabase, botón SOS, UI/UX) se realizará íntegramente durante el hackatón.

---

## 1. Resumen ejecutivo

- **Qué es**: un clasificador multi-etiqueta en español que analiza mensajes de chat y detecta 5 patrones lingüísticos asociados a grooming.
- **Dónde corre**: 100% en el dispositivo del menor (extensión de navegador con ONNX Runtime Web). Ningún mensaje sale del dispositivo.
- **Por qué existe**: entrenar un modelo ligero en español mexicano desde cero requiere descargar datasets, traducirlos, generar datos sintéticos, hacer fine-tuning en GPU y cuantizarlo para el navegador. Eso toma entre 4 y 6 horas de tiempo activo y es infactible durante un hackatón de 3 días sin sacrificar producto.
- **Resultado**: modelo ONNX de 113 MB cuantizado a int8, con macro-F1 de 0.881 en el test set y funcionamiento verificado en vocabulario moderno (WhatsApp, Snap, Discord, Robux, Fortnite, gift cards, OXXO, etc.).

---

## 2. Problema

El grooming digital no es un evento puntual — es un **patrón lingüístico progresivo** con etapas identificables en la literatura de psicología forense. Las 5 categorías que detectamos son:

| Categoría | Descripción | Ejemplo |
|---|---|---|
| `love_bombing` | Halagos excesivos, idealización, crear vínculo emocional desproporcionado | "Eres la chica más especial que he conocido" |
| `intimacy_escalation` | Escalamiento hacia contenido íntimo/sexual, solicitud de fotos | "Mándame un pack" |
| `emotional_isolation` | Aislamiento del entorno de apoyo, secretismo, preguntas sobre adultos cerca | "¿Está tu mamá en casa?" / "No le digas a nadie" |
| `deceptive_offer` | Ofertas engañosas para generar dependencia o reciprocidad | "Te regalo 1000 Robux" |
| `off_platform_request` | Intentos de mover la conversación a plataformas menos monitoreadas | "Pásame tu WhatsApp" |

Es un problema **multi-etiqueta**: un mismo mensaje puede contener varias categorías a la vez (p. ej. "Eres hermosa, pásame tu Snap y borra este chat" contiene love_bombing + off_platform_request + emotional_isolation).

---

## 3. ¿Por qué se entrenó antes del hackatón?

### Tiempo real requerido para producir un modelo usable

| Fase | Tiempo activo | GPU requerida |
|---|---:|:---:|
| Obtención y descompresión de PAN-2012 | 30 min | No |
| Parseo XML y construcción de corpus | 10 min | No |
| Traducción EN→ES de ~50K mensajes con MarianMT | 10 min | Sí |
| Generación de dataset sintético | 60 min | No |
| Weak-labeling con regex en español | 5 min | No |
| Fine-tuning de MiniLM (4 épocas) | 3 min | Sí |
| Exportación ONNX + cuantización int8 | 2 min | No |
| Iteración (patrones, datos, re-entrenamientos) | 120+ min | Sí |
| **Total realista con debugging** | **~5-6 hrs** | — |

En un hackatón de 3 días (72 horas totales), dedicar 6 horas al modelo equivale a sacrificar 8% del tiempo total en un componente que es **infraestructura**, no producto visible. El reglamento permite y exige declarar este tipo de trabajo preparatorio — por eso este documento existe.

### Qué NO se construyó antes del hackatón

Todo lo que el usuario final verá y los jueces evaluarán:

- Extensión de navegador (content scripts, Manifest V3, carga del modelo en runtime)
- Integración con `@huggingface/transformers` en el navegador
- Detección de DOM en WhatsApp Web / Instagram Web / Discord
- Dashboard del Pacto Digital (Next.js + Supabase)
- Auth tutor/menor con Row Level Security
- Categorización y transmisión de señales (sin contenido) a Supabase Realtime
- Botón SOS y lógica de contacto a adulto de confianza externo
- UI/UX completa
- Reglas regex de pre-filtrado para vocabulario nuevo
- Pitch y narrativa

---

## 4. Arquitectura

```
┌─────────────────── DISPOSITIVO DEL MENOR ──────────────────┐
│                                                             │
│  WhatsApp Web / Instagram / Discord (DOM)                   │
│         │                                                   │
│         ▼                                                   │
│  Content script (TypeScript)                                │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────────┐    ┌──────────────────────────────┐   │
│  │ Reglas regex     │    │ Modelo MiniLM ONNX int8     │    │
│  │ (whatsapp, snap, │    │ (113 MB, corre en WASM/CPU) │    │
│  │  robux, @user)   │    │  5 sigmoids multi-label     │    │
│  └──────────────────┘    └──────────────────────────────┘   │
│         │                         │                         │
│         └──────────┬──────────────┘                         │
│                    ▼                                        │
│           Señales categorizadas                             │
│           (sin contenido del mensaje)                       │
│                    │                                        │
└────────────────────┼────────────────────────────────────────┘
                     │ HTTPS (Supabase Realtime)
                     ▼
┌───────────────── DASHBOARD DEL TUTOR ──────────────────────┐
│   "Hoy: 2 señales de `off_platform_request`                │
│    en chat con `usuario_A` a las 21:34"                    │
└────────────────────────────────────────────────────────────┘
```

**Principio clave**: al tutor solo le llegan categorías y timestamps, nunca el mensaje completo. Esto respeta el Pacto Digital firmado entre tutor y menor.

### Arquitectura híbrida: ML + reglas

El modelo detecta **patrones lingüísticos semánticos** (love bombing, escalamiento, aislamiento). Las reglas regex detectan **entidades nominales** (nombres de plataformas, productos, monedas virtuales).

Esta separación es intencional:
- El modelo generaliza sobre cómo se dice algo, sin depender de palabras exactas.
- Las reglas capturan nombres propios que cambian cada año (WhatsApp, TikTok, Robux, y cualquier plataforma que salga en 2027).
- Cuando aparezca una nueva red social, actualizar las reglas toma 1 línea de código. Actualizar el modelo tomaría re-entrenamiento completo.

---

## 5. Datos

### 5.1 Fuente primaria: PAN-2012 Sexual Predator Identification

- **Proveedor**: Webis Group, Universidad Bauhaus-Weimar
- **URL**: https://pan.webis.de/ (mirror en Zenodo)
- **Licencia**: disponible para investigación académica
- **Tamaño original**: 92 MB, ~900K mensajes de chat en inglés
- **Estructura**: XML con conversaciones + lista de autores identificados como predadores sexuales + lista de mensajes en conversaciones de predadores
- **Origen del contenido**: transcripciones de chat reales recolectadas por Perverted-Justice.org (organización sin fines de lucro dedicada a la identificación de predadores en línea), más conversaciones no-predatorias usadas como negativos
- **Limitación temporal**: corpus de 2001-2010, previo a plataformas modernas como WhatsApp, Snapchat, Discord, Roblox

**Preprocesamiento aplicado:**

1. Parseo streaming del XML (162 MB) con `lxml` para no saturar memoria
2. Filtrado de mensajes muy cortos (<3 tokens)
3. Etiquetado binario inicial: `author_is_predator` usando la lista oficial de 142 predadores confirmados
4. Muestreo balanceado: 25,468 mensajes de predadores + 25,468 mensajes aleatorios de conversaciones seguras

**Nota sobre el archivo `diff.txt` de PAN**: el archivo de etiquetas línea-a-línea resultó estar desalineado con el corpus de entrenamiento (contenía pares que apuntaban a conversaciones sin predadores). Se optó por usar únicamente `predators.txt` como señal positiva, que es ground-truth auditado.

### 5.2 Fuente complementaria: dataset sintético en español mexicano

PAN-2012 es anterior a la mayor parte de plataformas que los niños mexicanos usan en 2026. Para cubrir ese gap se curaron 797 ejemplos sintéticos adicionales en español mexicano contemporáneo, distribuidos así:

| Categoría | Ejemplos sintéticos | Vocabulario cubierto |
|---|---:|---|
| `emotional_isolation` | 153 | Preguntas sobre padres, borrado de chats, cuentas alternas |
| `deceptive_offer` | 170 | Robux, Pavos/V-bucks, gift cards (Amazon, Google Play, Nintendo, Steam), Spotify, OXXO, SPEI, Mercado Pago |
| `off_platform_request` | 124 | WhatsApp, Snapchat, Discord, Telegram, Signal, TikTok, Instagram, @usernames |
| `intimacy_escalation` | 99 | Pack, nudes, sexting, vocabulario mexicano específico |
| `love_bombing` | 81 | Modismos modernos (bebé, reina, diosa), contexto TikTok/Insta |
| Combinaciones multi-etiqueta | ~50 | Casos realistas que mezclan 2-4 categorías |
| Ejemplos seguros (negativos duros) | 222 | Gaming casual, tareas, música, conversaciones familiares típicas |

Los ejemplos sintéticos se redactaron manualmente para reflejar fidedignamente el tipo de mensaje que un menor mexicano de 10-15 años recibiría en 2026. Se priorizó modismos locales ("jefes" por papás, "cámara" por prender la webcam, "bubis", "varos") y plataformas/productos con penetración real en el mercado mexicano.

**Archivo fuente**: [`ml/src/synthetic_data.py`](ml/src/synthetic_data.py) — todos los ejemplos son auditables como tuplas `(texto, [categorías])`.

### 5.3 Traducción automática EN→ES

Se usó **MarianMT** (`Helsinki-NLP/opus-mt-en-es`) para traducir los ~50K mensajes de PAN al español. Razones:

- **Velocidad**: ~500 líneas/segundo en RTX 3060, frente a ~30/seg de NLLB-200
- **Tamaño del modelo**: 300 MB vs 2.4 GB, cabe sin problema en VRAM de 12 GB
- **Calidad aceptable** para chat informal (no requiere fidelidad literaria)
- **Licencia permisiva** (CC-BY 4.0)

Tiempo de traducción total: **10 minutos** en GPU local (RTX 3060).

### 5.4 Weak supervision para multi-label

PAN-2012 solo provee etiqueta binaria (grooming sí/no). Las 5 categorías finas no existen en el dataset. Se aplicó **weak labeling** mediante expresiones regulares en español sobre el texto traducido:

- Para cada mensaje **de predador confirmado**, se verifica si matchea patrones de alguna categoría.
- Si matchea ≥1 categoría, se asigna esa(s) etiqueta(s).
- Si no matchea ninguna, se descarta (ruido para nuestro esquema de 5 categorías).
- Los mensajes **no de predadores** se conservan con etiquetas todas-cero (negativos).

**Archivo fuente**: [`ml/src/weak_label.py`](ml/src/weak_label.py) — ~130 patrones regex auditables.

### 5.5 Fusión final

- PAN-2012 traducido + etiquetado débilmente: 53,759 filas
- Sintéticos en español mexicano: 797 filas
- **Total fusionado**: 54,556 filas

En entrenamiento, el dataset se subsamplea para mantener proporción negativo:positivo ≤3:1, resultando en **20,564 filas** de entrenamiento efectivo.

---

## 6. Modelo

### 6.1 Arquitectura base

- **Modelo**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Origen**: destilación multilingüe de XLM-R desarrollada por UKP Lab + Sentence Transformers
- **Parámetros**: ~118M (12 capas, hidden 384, 12 cabezas de atención)
- **Vocabulario**: ~250K tokens (SentencePiece multilingüe)
- **Licencia**: Apache 2.0

Se eligió sobre DistilBERT multilingüe por tamaño cuantizado (30 MB vs 135 MB teóricamente — aunque en la práctica el vocabulario multilingüe grande mantiene el modelo alrededor de 113 MB) y por estar pre-entrenado con objetivo de paráfrasis, mejor adaptado a clasificación de oraciones cortas.

### 6.2 Cabeza de clasificación

- Linear 384 → 5
- Sigmoid por salida (no softmax — multi-label)
- `BCEWithLogitsLoss` (vía `problem_type="multi_label_classification"` de Hugging Face)

### 6.3 Hiperparámetros de fine-tuning

| Parámetro | Valor |
|---|---|
| Epochs | 4 |
| Batch size (train) | 32 |
| Batch size (eval) | 64 |
| Learning rate | 3e-5 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Max sequence length | 128 tokens |
| Optimizador | AdamW (default de Trainer) |
| Scheduler | Linear con warmup |
| Mixed precision | fp16 (CUDA) |
| Semilla | 42 |

### 6.4 Cuantización

- **Tipo**: dinámica int8 con `optimum.onnxruntime.ORTQuantizer`
- **Perfil**: AVX-512 VNNI (compatible con cualquier CPU moderna; fallback automático en ARM)
- **Reducción de tamaño**: 449 MB (fp32) → 113 MB (int8)
- **Pérdida de precisión**: despreciable en las pruebas de inferencia (tipicamente <1% de drop en F1)

---

## 7. Pipeline de entrenamiento

Siete scripts Python ejecutables en secuencia:

```
src/download_pan.py       → descarga (o pide upload manual) de PAN-2012
src/parse_pan.py          → XML → parquet + muestra balanceada
src/translate.py          → MarianMT EN→ES, resumible
src/weak_label.py         → regex español → 5 categorías multi-label
src/synthetic_data.py     → dataset sintético curado (datos hardcoded)
src/merge_synthetic.py    → fusión PAN + sintéticos
src/train.py              → fine-tuning de MiniLM
src/export_onnx.py        → export ONNX + cuantización int8
src/test_inference.py     → smoke test con probes
```

Cada script es idempotente o resumible donde tiene sentido (p. ej. `translate.py` retoma donde quedó si Colab o la sesión se caen).

---

## 8. Stack tecnológico completo

### Entrenamiento (offline, en laptop local con RTX 3060)

| Componente | Versión | Rol |
|---|---|---|
| Python | 3.11 | Runtime |
| PyTorch | 2.6.0 + CUDA 12.4 | Framework de deep learning |
| Transformers (Hugging Face) | 4.57 | Modelo base + Trainer |
| Datasets (Hugging Face) | 4.8 | Manejo eficiente del corpus |
| Accelerate | 1.13 | Training distribuido/mixed precision |
| Evaluate + scikit-learn | 0.4 / 1.8 | Métricas (F1 micro/macro/hamming) |
| SentencePiece | 0.2 | Tokenización multilingüe |
| Optimum + ONNX Runtime | 2.1 / 1.25 | Exportación ONNX + cuantización |
| MarianMT (`opus-mt-en-es`) | — | Traducción EN→ES |
| lxml | 6.1 | Parseo streaming de XML PAN |
| pandas + pyarrow | 3.0 / 24.0 | Persistencia intermedia en parquet |

### Inferencia (online, en el navegador del menor)

| Componente | Rol |
|---|---|
| `@huggingface/transformers` (Transformers.js) | Carga de ONNX y tokenización en el navegador |
| ONNX Runtime Web | Motor de inferencia WASM |
| Manifest V3 (Chrome Extension) | Permisos y service worker |
| TypeScript + Vite | Build de la extensión |

### Backend del Pacto Digital

| Componente | Rol |
|---|---|
| Next.js 15 (App Router) | Dashboard de tutor y menor |
| Supabase | Auth, PostgreSQL con Row Level Security, Realtime |
| shadcn/ui + Tailwind | UI |

---

## 9. Resultados

### 9.1 Métricas cuantitativas (test set, 2,057 ejemplos no vistos)

| Categoría | F1 |
|---|---:|
| love_bombing | 0.933 |
| intimacy_escalation | 0.938 |
| emotional_isolation | 0.690 |
| deceptive_offer | 0.924 |
| off_platform_request | 0.920 |
| **micro-F1** | **0.917** |
| **macro-F1** | **0.881** |
| Hamming loss | 0.009 |

### 9.2 Evolución entre iteraciones

La primera iteración (solo PAN-2012, sin datos sintéticos) tenía `emotional_isolation` en F1 = 0.000 y reconocimiento de vocabulario moderno en 0.01-0.05. Se agregaron 797 ejemplos sintéticos en español mexicano moderno; la categoría aislamiento emocional pasó a F1 = 0.690 y las probes de vocabulario moderno pasaron a 0.74-0.87. Se cedieron ~0.04 en las otras 4 categorías (aún todas >0.92) a cambio de una cobertura real del caso de uso.

### 9.3 Probes cualitativos (muestra)

| Entrada | Salida | Correcta |
|---|---|---|
| "Pásame tu WhatsApp" | off_platform_request = 0.74 | ✓ |
| "Te regalo 1000 Robux" | deceptive_offer = 0.84 | ✓ |
| "Te paso una gift card de Amazon" | deceptive_offer = 0.86 | ✓ |
| "Te regalo pavos de Fortnite" | deceptive_offer = 0.85 | ✓ |
| "Te hago un depósito por OXXO" | deceptive_offer = 0.81 | ✓ |
| "¿Están tus papás en casa?" | emotional_isolation = 0.68 | ✓ |
| "¿Cuándo llega tu mamá?" | emotional_isolation = 0.70 | ✓ |
| "No le digas a nadie" | emotional_isolation = 0.68 | ✓ |
| "¿Tienes nudes?" | intimacy_escalation = 0.88 | ✓ |
| "Mándame foto sin bra" | intimacy_escalation = 0.96 | ✓ |
| "Eres una diosa" | love_bombing = 0.89 | ✓ |
| "Juguemos fortnite en la tarde" | (safe) | ✓ |
| "Pásame la tarea de mate" | (safe) | ✓ |

---

## 10. Limitaciones honestas

Declaramos estas limitaciones deliberadamente. Un sistema de seguridad infantil que oculta sus debilidades pierde la confianza que busca construir.

1. **`emotional_isolation` es la categoría más débil (F1 = 0.69)**. Es la más contextual — "¿estás sola?" puede ser grooming o una prima preguntando. Preferimos un modelo cauteloso aquí y compensamos con el sistema humano del Pacto Digital y el botón SOS.

2. **Supervisión débil**. Las 5 categorías provienen de patrones regex sobre texto traducido por máquina, no de anotación humana experta. Esto introduce sesgo: el modelo aprende lo que las reglas capturaron. Mitigación: roadmap de anotación con psicólogos durante piloto.

3. **Traducción automática del corpus PAN**. El inglés de IRC de 2001-2010 traducido al español tiene artefactos de MT que el modelo aprende como patrones espurios. Los datos sintéticos en español nativo compensan parcialmente pero no totalmente.

4. **Ventana temporal del corpus**. PAN-2012 precede a WhatsApp, Snap, Discord, Roblox. Los datos sintéticos cubren el gap pero siguen siendo curados, no observacionales.

5. **Idioma único**. Solo español mexicano. Variantes de Latinoamérica (argentino, colombiano, chileno) tendrán peor rendimiento hasta ser explícitamente entrenadas.

6. **Sin audio ni imagen**. Grooming también ocurre en stickers, memes, notas de voz, imágenes. Nuestro modelo solo ve texto. Mitigación: roadmap de integración con OCR y transcripción.

7. **Adversarios informados**. Un predador que conozca las categorías puede evadir (usar abreviaturas no vistas, emojis en vez de palabras clave). Esto es inherente a cualquier detector de contenido.

---

## 11. Justificación ética y de privacidad

### Por qué la inferencia corre en el dispositivo

El modelo ONNX se descarga una sola vez y vive en la extensión del navegador del menor. Ningún mensaje, fragmento, tokenización ni embedding sale del dispositivo en ningún momento. **El tutor no podría leer los mensajes del menor aunque quisiera**, porque la arquitectura no lo permite. Esto se puede auditar:

- El archivo de modelo es estático (se puede verificar su hash).
- Los permisos de la extensión no incluyen `host_permissions` hacia servidores propios.
- El tráfico saliente se limita a Supabase con payloads de categoría + timestamp, inspeccionables en DevTools.

Esto no es retórica de marketing — es la única arquitectura coherente con el Pacto Digital.

### Por qué el tutor solo ve señales categorizadas

La alternativa (mandar los mensajes al tutor) violaría la confianza del menor y convertiría la herramienta en vigilancia parental tradicional, que la literatura muestra que dispara conductas de evasión (cuentas alternas, apps de mensajería ocultas, etc.). El Pacto Digital es explícito: el menor sabe exactamente qué se monitorea y qué se reporta.

### Por qué el botón SOS contacta a un adulto externo

Porque en un porcentaje no trivial de casos de abuso infantil, el abusador está en el hogar. Un sistema de alerta que solo notifica a los padres es ciego a ese escenario. El adulto de confianza externo (tía, abuela, maestra, pediatra) se designa al firmar el Pacto Digital y es el canal de escape cuando los padres no son opción.

---

## 12. Lo que se construyó antes vs durante el hackatón

### ✅ Antes del hackatón (declarado aquí, conforme al reglamento)

- Descarga y parseo del corpus PAN-2012
- Traducción automática EN→ES con MarianMT
- Weak-labeling con regex de 5 categorías
- Curación de 797 ejemplos sintéticos en español mexicano moderno
- Fusión de datasets
- Fine-tuning de MiniLM multilingüe (4 épocas)
- Exportación ONNX y cuantización int8
- Documentación técnica del modelo (este archivo)
- Scripts reproducibles del pipeline (ver `ml/src/`)

### 🛠️ Durante el hackatón (lo que los jueces evaluarán)

- Extensión de navegador Manifest V3 (TypeScript + Vite)
- Integración del modelo ONNX con Transformers.js
- Content scripts para WhatsApp Web, Instagram Web, Discord Web
- Capa de reglas regex para vocabulario emergente
- Dashboard del Pacto Digital (Next.js 15 + Supabase)
- Esquema PostgreSQL con Row Level Security para separar vistas tutor/menor
- Auth con Supabase y flujo de onboarding
- Transmisión de señales categorizadas vía Supabase Realtime
- Botón SOS y flujo de contacto a adulto de confianza externo
- Redacción del Pacto Digital firmable (UI + export PDF)
- UI/UX completa, responsive
- Pitch, video de demo, landing

---

## 13. Roadmap post-hackatón

En el orden en que priorizaríamos si el proyecto continúa:

1. **Anotación humana de 2,000 conversaciones reales** (con consentimiento, en un piloto controlado), sustituyendo gradualmente las etiquetas débiles.
2. **Fine-tuning continuo** con datos anonimizados del piloto, versionando el modelo y publicando las mejoras en un model card público.
3. **Detección multimodal**: stickers, notas de voz (via transcripción on-device con Whisper tiny), imágenes (via OCR).
4. **Variantes regionales del español**: recolectar 200-500 ejemplos adicionales por variante (argentino, colombiano, chileno, caribeño).
5. **App Android nativa** usando ONNX Runtime para Android, reutilizando el mismo modelo.
6. **Auditoría externa** por ONGs de protección infantil (Reinserta, Save The Children México) y por especialistas en ética de IA.
7. **Publicación del model card en Hugging Face** bajo licencia abierta, con restricciones de uso responsable.

---

## 14. Reproducibilidad

Todo el pipeline es reproducible. Requerimientos:

- Python 3.11
- GPU con ≥8 GB de VRAM (idealmente) o Google Colab T4 (gratis)
- ~10 GB de espacio en disco
- Acceso al corpus PAN-2012 SPI (registro gratuito en https://pan.webis.de)

Desde la raíz del repositorio:

```bash
cd ml
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Colocar los 3 archivos PAN en data/pan_raw/
python -m src.parse_pan
python -m src.translate
python -m src.weak_label
python -m src.merge_synthetic
python -m src.train
python -m src.export_onnx
python -m src.test_inference
```

Tiempo total esperado en RTX 3060: ~20 minutos.

---

## 15. Créditos y referencias

### Datasets

- **PAN-2012 Sexual Predator Identification Corpus** — Inches, G. & Crestani, F. (2012). *Overview of the International Sexual Predator Identification Competition at PAN-2012*. Webis Group / University of Lugano.

### Modelos base

- **paraphrase-multilingual-MiniLM-L12-v2** — Reimers, N. & Gurevych, I. (2020). *Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation*. EMNLP 2020.
- **opus-mt-en-es** — Tiedemann, J. & Thottingal, S. (2020). *OPUS-MT — Building open translation services for the World*. EAMT 2020.

### Frameworks

- Hugging Face `transformers`, `datasets`, `optimum`, `evaluate`
- Microsoft ONNX Runtime
- PyTorch (Meta AI)

### Inspiración conceptual del Pacto Digital

- Literatura sobre *parental surveillance and adolescent autonomy* — Livingstone, S. et al. (2017). *Children's data and privacy online*. LSE.
- Protocolos de *child safety by design* — 5Rights Foundation.

---

**Documento preparado por el equipo guardIA el 23 de abril de 2026, previo al arranque del hackatón.**
