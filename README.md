# Indic Transcribe

Upload a `.wav` file in Hindi, English or another Indian language, get its transcript,
and run the transcript through an LLM with your own prompt (e.g. to summarize it).
You choose the transcription engine:

- **Whisper**: runs locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Free; a GPU helps.
- **Sarvam AI**: calls the [Sarvam speech-to-text API](https://docs.sarvam.ai). Needs an API key.

For the LLM step you choose between Sarvam, Groq and Gemini (free tiers) or a local open
model with Ollama. `backend/process.py` runs the same transcribe → LLM pipeline from the
command line.

`backend/live.py` does the same live, from the microphone, in ~15 s chunks.

`backend/evaluate.py` tests either engine on your own dataset and reports WER and CER.
Two variants test Hinglish data in English letters (`evaluate_romanized.py`) or in
English (`evaluate_translated.py`).

## How it fits together

```
browser (public/)  →  Node server (server.js, :3000)  →  Python API (backend/app.py, :8000)
                                                              ├─ transcribers.py → Whisper or Sarvam
                                                              └─ llm.py          → Sarvam, Groq, Gemini or Ollama
```

The Node server serves the web page and forwards `/api/...` requests to the Python API.
All speech-to-text code lives in `backend/transcribers.py`, and the LLM step in `backend/llm.py`.

```
├── server.js              Node web server
├── package.json
├── public/
│   ├── index.html         the page
│   └── app.js             upload + display logic
└── backend/
    ├── app.py             FastAPI endpoints
    ├── transcribers.py    WhisperTranscriber, SarvamTranscriber
    ├── llm.py             sends a transcript + prompt to an LLM
    ├── prompt.txt         the default LLM prompt (edit freely)
    ├── process.py         command line: transcribe files, then run the LLM
    ├── live.py            live: transcribe the microphone in ~15 s chunks, LLM updates as it goes
    ├── prompts/           prompt templates for live.py (one file per mode)
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

## LLM step

After transcription, the transcript goes to an LLM with a prompt you control.

**Providers** (all use the same OpenAI-compatible chat API, so switching is one setting):

| Provider | Default model | Cost | Setup in `backend/.env` |
|---|---|---|---|
| `groq` | `llama-3.3-70b-versatile` | Free tier (rate-limited) | `GROQ_API_KEY` from [console.groq.com](https://console.groq.com) |
| `gemini` | `gemini-3.8-flash` | Free tier (rate-limited) | `GEMINI_API_KEY` from [aistudio.google.com](https://aistudio.google.com) |
| `sarvam` | `sarvam-105b` | Paid credits | `SARVAM_API_KEY` (the same key as for transcription) |
| `ollama` | `gemma3` | Free, runs locally | Install [Ollama](https://ollama.com), then `ollama pull gemma3` |

Model names change over time. If a default stops working, set `LLM_MODEL` (or the Model box
on the page) to a current one from the provider's model list. With Ollama, any model you've
pulled works.

**The prompt** is `backend/prompt.txt`. `{transcript}` is replaced by the transcript and
`{language}` by the spoken language (e.g. "Hindi"). If the prompt has no `{transcript}`, the
transcript is appended at the end. The default asks for an English summary.

**Settings** (`backend/.env`): `LLM_PROVIDER`, `LLM_MODEL` (empty = provider default),
`LLM_TEMPERATURE` (default 0.3; lower is more consistent), `LLM_MAX_TOKENS` (default 2048,
the maximum length of the answer).

**On the web page**, the LLM card shows all of these. Edits there apply to that page; to
change the defaults, edit `.env` and `prompt.txt` and restart the API. "Run automatically
after transcription" sends each new transcript straight to the LLM. **Run LLM** reruns it on
the same transcript, so you can try prompt changes without transcribing again.

**From the command line** (also works in Colab):

```bash
cd backend
python process.py recording.wav
python process.py clips/ --engine whisper --language hi --provider groq
python process.py clips/ --provider ollama --model gemma3 --prompt my_prompt.txt --temperature 0
```

Results are printed and appended to `outputs.jsonl`, one JSON object per file (`file`,
`language`, `transcript`, `llm_output`, `provider`, `model`). If the LLM step fails, the
transcript is still saved, with an `error` field.

## Live mode

`backend/live.py` transcribes the microphone as you speak and periodically sends the
conversation to an LLM to summarize it, brainstorm on it, or check its facts. It runs in the
terminal, on your own machine (Colab can't use your microphone).

```bash
cd backend
pip install -r requirements.txt          # adds sounddevice; on Linux also: sudo apt install libportaudio2
python live.py --language hi              # summarize mode, LLM settings from .env
python live.py --language hi --mode brainstorm --update-every 60
python live.py --engine sarvam --provider gemini
python live.py --input meeting.wav        # simulate live from a file (works in Colab too)
python live.py --help                     # all options
```

While it runs, type and press Enter: `s` updates the LLM output now, `? your question` asks
about the conversation so far, `q` (or Ctrl+C) stops. On stopping, the last chunk is
transcribed and the LLM writes a final version.

**How it works.** Four stages run in parallel, connected by queues, so a slow transcription
or LLM call never makes the recording lose audio:

```
microphone ──► segmenter ──► transcriber ──► LLM worker
               cuts ~15 s    one chunk at    updates every --update-every s of speech,
               at a pause    a time          on request, and at the end
```

- **Chunks are cut at pauses, not every N seconds.** A cut in the middle of a word makes
  it come out wrong on both sides. A chunk is cut at the first pause (`--pause`, 0.5 s)
  after `--min-chunk` (12 s). If nobody pauses, it's cut at the quietest moment before
  `--max-chunk` (25 s). `--cut fixed --chunk-seconds 15` cuts at fixed times instead, for
  comparison. Silent chunks are skipped, because Whisper invents text on silence.
- **Continuity:** Whisper gets the end of the previous transcript (`--context-chars`, 200) so
  names and spelling stay consistent across chunks. The language is detected once on the
  first chunk and then kept (`--redetect` detects it per chunk; `--language hi` skips detection).
- **Keeping up:** each chunk must be transcribed faster than it was spoken. If not, a warning
  appears. On a laptop CPU use `--whisper-model small` or `medium`, or `--engine sarvam`.
  `large-v3` needs a GPU to keep up. The full recording is saved, so you can re-transcribe it
  afterwards with `process.py` and `large-v3` for the best quality.
- **The LLM's context stays small.** Sending the whole transcript on every update would
  grow without limit and use up free API tiers quickly. Instead the LLM gets a running
  summary of everything older than `--recent-minutes` (4), plus the recent transcript word for
  word, split into what it has already seen and what's new. Older parts are folded into the
  summary by extra calls using `prompts/live_compress.txt`.

**Modes and prompts.** `--mode` picks `prompts/live_<mode>.txt`: `summarize` (default),
`brainstorm` or `verify`. Add a file, e.g. `prompts/live_minutes.txt`, and `--mode minutes`
works. `--prompt my_prompt.txt` uses any file instead. Placeholders filled in: `{summary}`,
`{recent}`, `{new}`, `{transcript}` (recent + new), `{previous_output}`, `{question}`,
`{language}`, `{status}`, `{elapsed}`. `verify` has no internet access: it checks claims against
the model's own knowledge and against what was said earlier, and marks doubtful ones as
"worth checking".

LLM options are the same as `process.py`: `--provider`, `--model`, `--temperature`,
`--max-tokens`, plus `--update-every` (seconds of speech between updates, 0 = only on request),
`--no-final`, and `--no-llm` to only transcribe.

**Saved in `sessions/<date_time>/`:** `audio.wav` (the recording), `transcript.jsonl` (one line
per chunk, with start and end times), `transcript.txt`, `llm.jsonl` (every LLM call, with the
exact prompt sent), `output.md` (the final LLM output) and `settings.json`.

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
