# Querly STT

Speech-to-text for Hindi, English, Hinglish and other Indian languages, with an LLM on top.

- **Live** (`querly-live/`): press record and talk. Your words appear on screen as you speak,
  transcribed by Sarvam's realtime API. Next to the transcript, an assistant powered by the
  backend's LLM keeps notes, brainstorms, checks facts or answers questions about the
  conversation, and writes a final version when you stop.
- **Upload a clip** (`public/`, `server.js`): upload a `.wav` file, transcribe it with Whisper
  (local) or Sarvam, and run the transcript through an LLM with your own prompt.
- **Evaluation** (`backend/evaluate*.py`): measure WER and CER of either engine on your own
  dataset, including romanized and English-translated comparisons for Hinglish.

## How it fits together

```
LIVE
browser (querly-live/static) ──WebSocket──► querly-live/live_server.py (:8080) ──WebSocket──► Sarvam realtime STT
                                                  │ each finished line
                                                  ▼
                                          backend/app.py (:8000) ── live_llm.py ── llm.py ──► Groq / Gemini / Sarvam / Ollama

UPLOAD A CLIP
browser (public/) ──► server.js (:3000) ──► backend/app.py (:8000) ── transcribers.py ──► Whisper or Sarvam
                                                                  └─ llm.py
```

- **querly-live** does the live transcription. The browser records the microphone, so
  nothing native needs installing. The server streams the audio to Sarvam and pushes
  partial and final text back to the page. Each finished line is also sent to the backend.
- **The backend** does all LLM work, for both live mode and uploaded clips. For live mode it
  keeps the conversation's context (a running summary plus the recent transcript), runs
  periodic updates, answers questions and writes the final version.
- If the backend isn't running, live transcription still works; the assistant just says
  it's unavailable.

```
├── README.md
├── querly-live/               LIVE MODE
│   ├── live_server.py         server: browser <-> Sarvam, and lines -> backend
│   ├── llm_bridge.py          sends lines to the backend, relays LLM results to the page
│   ├── static/index.html      the live page (transcript + assistant panel)
│   ├── mock_sarvam.py         a fake Sarvam socket, for testing with no key
│   ├── selftest.py            end-to-end check of the live path (no key needed)
│   ├── requirements.txt
│   └── .env.example           Sarvam key for live mode
├── backend/                   PYTHON API (LLM work, clip transcription, evaluation)
│   ├── app.py                 FastAPI endpoints (clips, LLM, live assistant)
│   ├── live_llm.py            live assistant: context, rolling summary, modes
│   ├── prompts/               live assistant prompts, one file per mode
│   ├── llm.py                 sends prompts to an LLM provider
│   ├── prompt.txt             default prompt for uploaded clips
│   ├── transcribers.py        WhisperTranscriber, SarvamTranscriber (clip mode)
│   ├── process.py             command line: transcribe files, then run the LLM
│   ├── evaluate.py            test against a dataset
│   ├── evaluate_romanized.py  ...comparing romanized (Latin-script) text
│   ├── evaluate_translated.py ...comparing English translations
│   ├── requirements.txt
│   └── .env.example           LLM keys, Whisper settings, Sarvam key for clips
├── server.js, package.json    UPLOAD-A-CLIP web server (Node)
└── public/                    the upload page
```

## Setup

You need **Python 3.10+**, and **Node 18+** only for the upload page.

On Windows, check Python in PowerShell with `python --version`. If nothing happens or the
Microsoft Store opens, install Python from <https://www.python.org/downloads/windows/> and
tick **"Add python.exe to PATH"**. If `python` still doesn't work, use `py -3` instead.

```bash
# One virtual environment for both Python parts, from the repo root
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt -r querly-live/requirements.txt

# Settings: two .env files, one per part
cp backend/.env.example backend/.env           # LLM keys (e.g. GROQ_API_KEY), Whisper settings
cp querly-live/.env.example querly-live/.env   # SARVAM_API_KEY for live mode

# Only for the upload page
npm install
```

**Keys:**
- **Sarvam:** for live mode, put it in `querly-live/.env`. Get one at
  <https://indus.sarvam.ai/key-management>; a new account gets ₹100 of free credits. The
  realtime endpoint bills per second of audio, so a 10-minute conversation is 600 billed
  seconds. To also use Sarvam for clips or as the LLM, add the same key to `backend/.env`.
- **LLM:** at least one provider in `backend/.env`, e.g. `GROQ_API_KEY` (free tier). See
  [LLM providers](#llm-providers).
- Don't paste keys into chats, screenshots or the repo. Both `.env` files are git-ignored.

Two Windows traps with `.env` files:
- **The file must be named `.env`, not `.env.txt`.** In Notepad, use *Save as*, set *Save as
  type* to **All Files** and type `.env`. Check with `dir .env` in PowerShell.
- No spaces around `=`, and no quotes around the key.

## Live mode

Two terminals, both with the virtual environment activated:

```bash
# terminal 1: the backend (LLM work)
cd backend
uvicorn app:app --port 8000

# terminal 2: live transcription
cd querly-live
python live_server.py
```

The browser opens <http://127.0.0.1:8080>. Click the round mic button (or press **space**)
and talk. The first time, the browser asks for microphone permission: click **Allow**. Use
Chrome or Edge.

### Transcription settings

- **Language**: leave it on *Hindi + Hinglish*. Measured against the live service, *Auto-detect*
  returns English guesses for the first ~0.7 s of every utterance (literally `, How are you?`
  for Hindi speech) before it settles, because the model hasn't heard enough audio to pick a
  language. Pinned to Hindi, the first partial arrives 0.34 s after you start speaking and is
  already correct. Use *Auto-detect* only if you switch languages a lot mid-conversation.
- **Script**: *Hinglish — Latin* (default) gives `Haan bhai, kal ki meeting mein hum real time
  transcription ship karenge`. *Devanagari* gives the same in native script. *Mixed script*
  turned out to write English words in Devanagari for Hindi speech (`रियल टाइम ट्रांसक्रिप्शन`),
  so it isn't the Hinglish you'd expect. *English translation* translates instead of transcribing.
- **Partials always arrive in native script**, whichever Script you pick; that's how the
  service works. While you speak you read Devanagari, and the line switches to Latin
  Hinglish when it's final. If that bothers you, set Script to *Devanagari*.
- In-progress guesses appear in bright white with a caret; when an utterance finishes, the
  final text replaces the guess and the line settles.
- **Copy** / **Save .txt** take the whole transcript so far.

### The assistant (LLM)

The panel on the right sends the transcript to the backend's LLM while you talk.

- **Mode**: what the LLM does. `summarize` keeps notes (overview, main points, decisions and
  action items). `brainstorm` suggests ideas, questions and risks. `verify` lists factual
  claims and flags doubtful ones. Each mode is a prompt file,
  `backend/prompts/live_<mode>.txt`; add a file (e.g. `live_minutes.txt`) and restart the
  backend to add a mode.
- **Updates**: every *Update every* seconds of speech (default 120) the LLM refreshes its
  output. **Update now** refreshes immediately; 0 means only on request.
- **Ask**: a question about the conversation so far, answered without changing the notes.
- **Final version**: when you stop, the LLM writes a final version. The page waits for it.
- **Provider, Model, Temperature, Max tokens**: empty boxes use the defaults in `backend/.env`.
- **Prompt**: the selected mode's template, editable. Edits apply to the next recording on
  this page; to change a mode permanently, edit its file.
- Settings are fixed while recording. Untick *Send the transcript to the LLM* to only transcribe.

**How the LLM sees a long conversation.** Sending the whole transcript on every update would
grow without limit and use up free API tiers quickly. Instead the LLM gets a running summary
of everything older than *Recent window* (4 minutes), plus the recent transcript word for word,
split into what it has already seen and what's new. Older parts are folded into the summary by
extra calls using `prompts/live_compress.txt`, so each request stays about the same size
however long you talk.

Placeholders available in the prompt files: `{summary}`, `{recent}`, `{new}`, `{transcript}`
(recent + new), `{previous_output}`, `{question}`, `{language}`, `{status}`, `{elapsed}`.
`verify` has no internet access: it checks claims against the model's own knowledge and
against what was said earlier, and marks doubtful ones as "worth checking".

**If something fails**: transcription never stops because of the LLM. If the backend can't
be reached, lines are kept and sent once it's back. If the backend restarts, its session is
rebuilt from the transcript so far. LLM errors appear as cards in the panel.

### Where live sessions are saved

Every recording creates `querly-live/sessions/YYYY-MM-DD_HHMMSS/` (git-ignored) containing:

- `audio.wav`: the full recording, 16 kHz mono. If the live transcript was rough, this can be
  re-transcribed offline with a bigger model, e.g. `backend/process.py` with Whisper `large-v3`.
- `transcript.txt`: the finished lines, one per line.
- `transcript.jsonl`: the same, with each line's `start` and `end` (seconds into
  `audio.wav`), detected language and session number. `session_gen` counts reconnects: after
  a dropped connection the utterance numbering restarts, so `(session_gen, utterance_idx)` is
  the unique key.
- `llm.jsonl`: every LLM result (updates, answers, final), with the exact prompt sent.
- `output.md`: the final LLM output.

### Latency you should expect

Measured on the live Sarvam service, streaming real Hinglish speech through this app (16 kHz,
50 ms frames, `stream_type=fast`, `silence_duration_ms=400`):

| | Measured |
|---|---|
| Connect to Sarvam | 0.05–0.11 s |
| First word on screen, after you start speaking | **0.34 s** (Hindi pinned) / 0.73 s (auto-detect) |
| Line finalises, after you stop speaking | **0.58–0.61 s** |

By comparison, transcribing ~15 s chunks with Whisper shows text 12–25 seconds late.

### Testing without a key

```bash
cd querly-live
python selftest.py
```

You should see `ALL 31 CHECKS PASSED`. It runs the whole live path against the fake Sarvam
socket, never contacts Sarvam and needs no key: the browser protocol, every byte of audio
reaching `audio.wav`, reconnecting after a dropped socket, a refused session producing a
readable error, and the stop path.

To click through the page with no key, run the fake socket in one terminal:

```bash
cd querly-live
python mock_sarvam.py
```

and the app in another (PowerShell shown; on macOS/Linux use `export NAME=value`):

```powershell
$env:SARVAM_WS_URL="ws://127.0.0.1:8765/speech-to-text-realtime/ws"
$env:SARVAM_API_KEY="dummy"
python live_server.py
```

**The fake socket doesn't listen to your audio.** It replays a fixed list of Hinglish lines,
which is enough to see the transcript and the assistant working. Afterwards, close that
terminal before running the real thing (the `$env:` lines stay set in it). To preview the
look with no audio at all, open <http://127.0.0.1:8080/?demo=1>.

### What has been tested, and what hasn't

- **Tested on the live Sarvam service with a real key:** connection and authentication, the
  exact query parameters, partial and final events, voice-activity timing, all four Script
  modes, Hindi-pinned vs auto-detect, `fast` vs `balanced`, and the stop path.
- **Tested automatically:** `selftest.py` (31 checks) covers the live path. The audio
  worklet's resampling is unit-tested at 48, 44.1 and 16 kHz.
- **LLM assistant:** tested end to end with the real backend, the fake Sarvam socket and a
  stand-in LLM server, both through a browser-like client and in a real browser with a fake
  microphone. That covered automatic updates, Update now, questions, the final version,
  compression of older transcript, a backend restart mid-session, and an unreachable backend
  or LLM. It has **not** yet been run with a real LLM provider.
- **Not tested:** your microphone, the browser's permission prompt and your own voice. Each
  of these shows a clear message on screen if it fails.

### Troubleshooting

| What you see | What it means |
|---|---|
| Red banner "No API key yet" | `querly-live/.env` is missing, misnamed (`.env.txt`), or the server started before you saved it. Fix it, then restart `python live_server.py`. |
| "Sarvam rejected the API key (HTTP 403)" | The key is wrong, or has a stray space or quote in `.env`. Make a fresh key and paste it again. |
| "Sarvam refused the session (rate limit, quota exhausted, or invalid API key)" | You're out of credits, or the key was revoked. Check the dashboard billing page. |
| Nothing happens when you click | The browser blocked the mic. Click the padlock in the address bar → Microphone → Allow, then reload. On Windows also check Settings → Privacy & security → Microphone → "Let desktop apps access your microphone". |
| Status says "no key" but you set one | You edited `.env` while the server was running. Restart it. |
| Assistant says "unavailable" | The backend isn't running, or not on port 8000. Start it (`uvicorn app:app --port 8000` in `backend/`), then reload the page. |
| Assistant: "no LLM provider set up" | No LLM key in `backend/.env`. Add one (e.g. `GROQ_API_KEY`) and restart the backend. |
| Assistant error cards with an LLM error | The provider refused (wrong key, rate limit, retired model name). Transcription continues; fix `backend/.env` for the next recording. |
| "Address already in use" | Another program uses the port. Run `python live_server.py --port 8123` and open the URL it prints. |
| Text appears but very late | Set Speed to *Fastest*. On a slow connection to Sarvam, partials back up. |
| Nothing in the transcript but the recording saved | The key or quota failed mid-session; the audio in `sessions/` can be re-transcribed later. |
| Microphone picks up your speakers | Use headphones, or disable "listen to this device" in your sound settings. Echo cancellation is on by default. |

### Options

- `python live_server.py --port 8123`: a different port (default 8080).
- `--no-browser`: don't open the browser automatically.
- `--host 0.0.0.0`: let other devices on your network open the page. The audio then crosses
  the network unencrypted; keep it on localhost unless you need it.
- `BACKEND_URL` environment variable: where the backend runs, if not `http://127.0.0.1:8000`.
- Transcription model: `saaras:v3-realtime` is the default. `saaras:v4` is also available on
  this endpoint with the same protocol; to try it, add `"model": "saaras:v4"` to the `start`
  message in `static/index.html` (search for `stream_type`).

## Upload a clip

Two terminals, with the virtual environment activated in the first:

```bash
# terminal 1
cd backend
uvicorn app:app --port 8000

# terminal 2 (repo root)
npm start
```

Open <http://localhost:3000>, pick a `.wav` file, an engine and a language (or Auto-detect),
and click **Transcribe**. If the LLM is set up, the transcript then goes through it with the
prompt from `backend/prompt.txt`. **Run LLM** reruns it on the same transcript, so you can
try prompt changes without transcribing again. The backend serves both this page and live
mode, so one backend is enough for both.

- **Engines:** **Whisper** runs locally with
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper); it's free, and a GPU helps.
  **Sarvam** calls the [Sarvam speech-to-text API](https://docs.sarvam.ai) and needs
  `SARVAM_API_KEY` in `backend/.env`.
- **The first Whisper request downloads the model** (about 3 GB for `large-v3`). On a laptop
  without a GPU, set `WHISPER_MODEL=small` or `medium` in `backend/.env`.
- **NVIDIA GPU (including Google Colab):** also run `pip install nvidia-cublas-cu12`.
  faster-whisper needs CUDA 12's cuBLAS library, which many systems don't provide, and loads
  it from this package automatically. Without it, the GPU fails with
  `Library libcublas.so.12 is not found`. A newer driver (e.g. one reporting CUDA 13) is fine.
- **The prompt** is `backend/prompt.txt`. `{transcript}` is replaced by the transcript and
  `{language}` by the spoken language. If the prompt has no `{transcript}`, the transcript is
  appended at the end.

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

## LLM providers

Live mode and the upload page share these, configured in `backend/.env`. All use the same
OpenAI-compatible chat API, so switching is one setting.

| Provider | Default model | Cost | Setup in `backend/.env` |
|---|---|---|---|
| `groq` | `llama-3.3-70b-versatile` | Free tier (rate-limited) | `GROQ_API_KEY` from [console.groq.com](https://console.groq.com) |
| `gemini` | `gemini-3.8-flash` | Free tier (rate-limited) | `GEMINI_API_KEY` from [aistudio.google.com](https://aistudio.google.com) |
| `sarvam` | `sarvam-105b` | Paid credits | `SARVAM_API_KEY` |
| `ollama` | `gemma3` | Free, runs locally | Install [Ollama](https://ollama.com), then `ollama pull gemma3` |

**Settings:** `LLM_PROVIDER`, `LLM_MODEL` (empty = provider default), `LLM_TEMPERATURE`
(default 0.3; lower is more consistent), `LLM_MAX_TOKENS` (default 2048, the maximum length of
an answer). Model names change over time: if a default stops working, set `LLM_MODEL` (or the
Model box on either page) to a current one from the provider's model list. With Ollama, any
model you've pulled works.

## Languages

Live mode (Sarvam realtime): English, Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati,
Kannada, Malayalam, Punjabi and Odia on the page, plus Hinglish in Latin script.

Upload a clip: the same eleven languages. Odia works only with Sarvam; Whisper doesn't support
it. With **Auto-detect**, Whisper chooses only among these languages; its default
~100-language detection often mislabels Hindi as Urdu and outputs Urdu script.

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
cd backend

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

### Hinglish: romanized and English evaluation

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
