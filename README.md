# Indic Transcribe

Upload a `.wav` file in Hindi, English or another Indian language and get its transcript.
You choose the engine:

- **Whisper**: runs locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Free; a GPU helps.
- **Sarvam AI**: calls the [Sarvam speech-to-text API](https://docs.sarvam.ai). Needs an API key.

`backend/evaluate.py` tests either engine on your own dataset and reports WER and CER.
Two variants test Hinglish data in English letters (`evaluate_romanized.py`) or in
English (`evaluate_translated.py`).

## How it fits together

```
browser (public/)  →  Node server (server.js, :3000)  →  Python API (backend/app.py, :8000)
                                                              └─ transcribers.py → Whisper or Sarvam
```

The Node server serves the web page and forwards `/api/...` requests to the Python API.
All speech-to-text code lives in `backend/transcribers.py`.

```
├── server.js              Node web server
├── package.json
├── public/
│   ├── index.html         the page
│   └── app.js             upload + display logic
└── backend/
    ├── app.py             FastAPI endpoints
    ├── transcribers.py    WhisperTranscriber, SarvamTranscriber
    ├── evaluate.py        test against a dataset
    ├── evaluate_romanized.py   ...comparing romanized (Latin-script) text
    ├── evaluate_translated.py  ...comparing English translations
    ├── requirements.txt
    └── .env.example
```

## Setup

Requires Python 3.10+ and Node 18+.

```bash
# Python backend
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then edit: model size, Sarvam key

# Node server (from the repo root)
cd ..
npm install
```

## Run

Two terminals:

```bash
# terminal 1
cd backend && source .venv/bin/activate
uvicorn app:app --port 8000

# terminal 2
npm start
```

Open http://localhost:3000, pick a file, an engine and a language (or Auto-detect), and click **Transcribe**.

The first Whisper request downloads the model (about 3 GB for `large-v3`). On a laptop
without a GPU, set `WHISPER_MODEL=small` or `medium` in `backend/.env`.

**Using an NVIDIA GPU (including Google Colab):** also run `pip install nvidia-cublas-cu12`.
faster-whisper needs CUDA 12's cuBLAS library, which many systems don't provide, and
loads it from this package automatically. Without it, the GPU fails with
`Library libcublas.so.12 is not found`. A newer driver (e.g. one reporting CUDA 13) is fine;
drivers run older CUDA libraries.

## Languages

English, Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, and Odia.
Odia works only with Sarvam; Whisper doesn't support it.

With **Auto-detect**, Whisper chooses only among these languages. Its default ~100-language
detection often mislabels Hindi as Urdu and outputs Urdu script.

## Testing on your dataset

The dataset is two files:

- **`transcripts.csv`**: one clip per line, `path/to/clip.wav, transcript`. A header line is
  fine, and transcripts may contain commas (only the first comma separates the fields).
  ```csv
  speaker1/clip001.wav, मैं कल दिल्ली जा रहा हूँ।
  speaker1/clip002.wav, The train leaves at nine.
  ```
- **`clips.tar.gz`**: the audio. It's extracted once into `clips/` next to the archive, and
  later runs reuse that folder. You can also pass an already-extracted folder.

Paths in the CSV are matched inside the archive, even if the archive adds or drops a
top-level folder (e.g. the CSV says `speaker1/clip001.wav` and the archive holds
`data/speaker1/clip001.wav`). CSV lines with no matching audio are reported and skipped.

```bash
cd backend && source .venv/bin/activate

python evaluate.py --csv ../data/transcripts.csv --audio ../data/clips.tar.gz --engine whisper --limit 10   # quick trial
python evaluate.py --csv ../data/transcripts.csv --audio ../data/clips.tar.gz --engine whisper
python evaluate.py --csv ../data/transcripts.csv --audio ../data/clips.tar.gz --engine sarvam
python evaluate.py --csv ../data/transcripts.csv --audio ../data/clips.tar.gz --engine whisper --language hi
```

`--language hi` tells the engine the language instead of auto-detecting it. Running with and
without it shows how much error comes from language detection.

Keep the dataset outside the repo (as `../data` above) so audio isn't committed.

Example output (numbers are illustrative):

```
           files       WER       CER     RTF
ALL           40    23.10%     9.85%    0.41

Per-file results saved to results.csv
```

`results.csv` has one row per clip: its CSV path, reference, model output, both normalized,
WER, CER and timing. Use it to see where the errors come from.

### What the metrics mean

| Metric | Meaning |
|---|---|
| **WER** | Word error rate: (substituted + deleted + inserted words) ÷ words in the reference. Lower is better. |
| **CER** | The same at the character level. Fairer for Tamil, Telugu, Malayalam and Kannada, whose long words make WER harsh: one wrong suffix fails the whole word. |
| **RTF** | Real-time factor: processing time ÷ audio length. Below 1 means faster than real time. |

WER and CER are computed over the whole dataset (total errors ÷ total reference words), so
a long clip counts more than a short one.

Before scoring, both texts are normalized so formatting isn't counted as an error. Unicode
is made consistent, punctuation (including `।`) is removed, native digits become `0-9`, and
English is lower-cased. Two differences still count as errors:

- numbers written as digits vs words (`20` vs `बीस`)
- English words in Latin vs Devanagari script in mixed speech (`apply` vs `अप्लाई`)

Check `results.csv` before concluding that one engine is worse.

## Hinglish: romanized and English evaluation

Hinglish references mix scripts: the same English word is `slide` in one sentence and
`स्लाइड` in another. `evaluate.py` counts that as an error even when the model heard it right.
Two variants of the evaluation compare in English letters instead. Both take the same options
as `evaluate.py`.

**`evaluate_romanized.py`**: romanizes the reference *and* the model output (Hindi written in
English letters), then scores them.

```bash
python evaluate_romanized.py --csv transcripts.csv --audio clips.tar.gz --engine whisper --language hi
python evaluate_romanized.py --csv transcripts.csv --audio clips.tar.gz --engine whisper --language hi --romanizer sarvam
```

- `--romanizer uroman` (default) is free and local; `sarvam` uses Sarvam's transliteration
  API and gives more natural spelling.
- Romanized Hindi has no standard spelling, so repeated letters are collapsed on both sides
  (`jaa` = `ja`).
- English words written in Devanagari come out close but not identical (`स्लाइड` → `slaid`).
  CER handles that better than WER.

**`evaluate_translated.py`**: translates the reference into English, has the model translate
the speech into English (Whisper `task="translate"`, Sarvam `mode="translate"`), and compares
the two.

```bash
python evaluate_translated.py --csv transcripts.csv --audio clips.tar.gz --engine whisper --language hi
python evaluate_translated.py --csv transcripts.csv --audio clips.tar.gz --engine sarvam --language hi --translator nllb
```

- References are translated with `--translator`:
  - `google` (default): Google Translate. Free, no key, needs internet. Its free endpoint
    rate-limits, so requests are spaced 1 s apart and retried with growing waits. If it
    still gives up, rerun the same command; finished translations are kept.
  - `nllb`: Meta's NLLB-200 model, run locally and offline after a one-time ~2.5 GB download.
    Fast on a GPU such as Colab's, slow on a CPU. `transformers` and `sentencepiece` come from
    `requirements.txt`; it also needs **torch**, which Colab already has. On your own machine,
    install torch from [pytorch.org](https://pytorch.org); the right build depends on your GPU.
  - `sarvam`: Sarvam's translation API. Needs `SARVAM_API_KEY`.
- Each reference is translated once and saved next to your CSV, e.g.
  `transcripts_english_google.csv`. Check it and fix bad lines; later runs use your
  corrections. If you already have English references, save them under that name
  (`path,english text`) and nothing is translated.
- Also reports **BLEU** and **chrF** (0–100, higher is better), the standard translation
  metrics. A sentence can be translated correctly in many ways, so compare engines with these
  rather than WER.

**About `--language`:** it's the language *spoken* in the clips, so use `hi` for Hindi audio
in both scripts. The English output comes from romanizing or from translate mode.
`--language en` tells Whisper the audio is English. On Hindi speech it then produces a
translation-like English output without using its translation mode, which matches neither
reference well.
