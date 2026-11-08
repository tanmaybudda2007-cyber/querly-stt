# Querly Live — Windows 11 setup

Press a button, talk, and your words appear on screen as you speak, like lyrics.
Hindi, English and Hinglish (mixed), auto-detected.

You need one thing before this works: a Sarvam API key. Everything else is done.

---

## 1. Get the API key (2 minutes)

1. Open <https://indus.sarvam.ai/key-management> and sign up (Google sign-in works).
2. Create an API key and copy it. It looks like a long random string.
3. Keep it somewhere private. Don't paste it into chat, a screenshot or a repo.

A new account gets ₹100 of free credits that never expire, which is a lot of talking
for testing. The realtime endpoint bills per second of audio sent, so a 10-minute
conversation is 600 billed seconds.

## 2. Put the files somewhere permanent

Unzip this folder to somewhere like `C:\Users\<you>\querly-live`.
Don't run it from inside the .zip — Windows treats that as a temp folder.

You should see:

```
querly-live\
  app.py               the server: browser <-> Sarvam
  static\index.html    the page you look at
  mock_sarvam.py       a fake Sarvam socket, for testing with no key
  selftest.py          checks the whole path works
  requirements.txt
  .env.example         copy this to .env
  RUN_WINDOWS.md       this file
```

## 3. Install Python and the libraries

Check Python first. Open **PowerShell** (press Win, type `powershell`) and run:

```powershell
python --version
```

You want 3.10 or newer. If you see nothing, or the Microsoft Store opens, install
Python from <https://www.python.org/downloads/windows/> and tick **"Add python.exe to
PATH"** during install. If `python` still doesn't work, use `py -3` instead of `python`
in every command below.

Then install the three libraries:

```powershell
cd C:\Users\<you>\querly-live
python -m pip install -r requirements.txt
```

That's it — no ffmpeg, no PortAudio, no sounddevice. The browser does the audio.

## 4. Add your key

Copy `.env.example` to `.env` in the same folder, then open `.env` in Notepad and paste
your key after `=`:

```
SARVAM_API_KEY=your_key_here
```

Two Windows traps, both common:

- **The file must be named `.env`, not `.env.txt`.** In Notepad use *Save as*, set
  *Save as type* to **All Files**, and type `.env` with the quotes if needed.
  Check with `dir .env` in PowerShell — if it shows `.env.txt`, rename it.
- No spaces around `=`, and no quotes around the key.

## 5. Check it works before spending anything

```powershell
python selftest.py
```

You should see `ALL 31 CHECKS PASSED`. This runs the whole pipeline against the fake
socket — it never contacts Sarvam, so it uses no credits and doesn't need a key yet.
If this passes, your install is good and any remaining problem is the key or the mic.

## 6. Run it

```powershell
python app.py
```

Your browser opens at <http://127.0.0.1:8000>. Click the round mic button (or press
**space**) and start talking. The first time, Chrome/Edge asks for microphone
permission — click **Allow**.

Use Chrome or Edge. Both handle the 16 kHz audio graph properly.

## 7. Using it

- **Space** or the round button starts and stops.
- **Language**: leave on *Hindi + Hinglish*. I measured this against the live service and
  it is not a close call — with *Auto-detect* the first ~0.7 s of every utterance comes
  back as English guesses (literally `, How are you?` for Hindi speech) before it settles,
  because the model has not heard enough audio to pick a language yet. Pinned to Hindi,
  the first partial arrives 0.34 s after you start speaking and is already correct. Switch
  to *Auto-detect* only if you are switching languages a lot mid-conversation.
- **Script**: *Hinglish — Latin* (the default) gives `Haan bhai, kal ki meeting mein hum
  real time transcription ship karenge`. *Devanagari* gives the same in native script.
  *Mixed script* turned out to transliterate English words into Devanagari for Hindi speech
  (`रियल टाइम ट्रांसक्रिप्शन`), so it is not the Hinglish you want. *English translation*
  translates instead of transcribing.
- **Partials always arrive in native script**, whichever Script you pick — that is how the
  service works ("partials are always plain transcription"). So while you are speaking you
  read Devanagari, and the line flips to Latin Hinglish the moment it finalises. If that
  flip bothers you, set Script to *Devanagari* and everything stays in one script.
- Partials (in-progress guesses) appear in bright white with a caret; each finished
  utterance replaces the guess with the final text and the line settles down.
- **Copy** / **Save .txt** take everything said so far.
- Every session is also saved on disk — see below.

---

## Testing with no key (or to check the page)

The fake socket replays a fixed list of Hinglish lines. **It does not listen to your
audio** — it exists to prove the page, the server and the audio path work.

Open a second PowerShell window:

```powershell
cd C:\Users\<you>\querly-live
python mock_sarvam.py
```

Then in the first window:

```powershell
$env:SARVAM_WS_URL="ws://127.0.0.1:8765/speech-to-text-realtime/ws"
$env:SARVAM_API_KEY="dummy"
python app.py
```

Talk (or just watch): the canned lines type themselves out. To preview the look with
no audio at all, open <http://127.0.0.1:8000/?demo=1>.

Close the second window to stop the fake socket, and start a fresh PowerShell window
before running the real thing again (that `$env:` line would otherwise still be set).

---

## Troubleshooting

| What you see | What it means |
|---|---|
| Red banner "No API key yet" | `.env` is missing, misnamed (`.env.txt`), or the server started before you saved it. Fix it, then restart `python app.py`. |
| "Sarvam rejected the API key (HTTP 403)" | The key is wrong, or has a stray space or quote in `.env`. Make a fresh key and paste it again. |
| "Sarvam refused the session (rate limit, quota exhausted, or invalid API key)" | You're out of credits, or the key was revoked. Check the dashboard billing page. |
| Nothing happens when you click | The browser blocked the mic. Click the padlock icon in the address bar → Microphone → Allow, then reload. Also check Windows Settings → Privacy & security → Microphone → "Let desktop apps access your microphone". |
| Status says "no key" but you set one | You edited `.env` while the server was running. Restart it. |
| "Address already in use" | Port 8000 is taken. Run `python app.py --port 8123` and open the URL it prints. |
| Text appears but is very late | Check the Speed selector is on *Fastest*. On a slow connection to Sarvam, partials back up. |
| Nothing in the transcript but the recording saved | The key or quota failed mid-session; the audio in `sessions\` can be re-transcribed later. |
| Microphone picks up your speakers | Use headphones, or in Windows Sound settings disable "listen to this device". Echo cancellation is on by default. |

## Where things are saved

Every session creates `sessions\YYYY-MM-DD_HHMMSS\` containing:

- `audio.wav` — the full recording, 16 kHz mono. This is your safety net: if the live
  transcript was rough, this file can be re-transcribed offline with a bigger model.
- `transcript.txt` — the finished lines, one per line.
- `transcript.jsonl` — the same, with timestamps, detected language and session number.

`transcript.jsonl`'s `session_gen` counts reconnects: if the connection dropped and came
back, the utterance numbering restarts, so the pair `(session_gen, utterance_idx)` is the
unique key.

## Latency you should expect

Measured on the live Sarvam service, streaming real Hinglish speech through this exact
app (16 kHz, 50 ms frames, `stream_type=fast`, `silence_duration_ms=400`):

| | Measured |
|---|---|
| Connect to Sarvam | 0.05–0.11 s |
| First word on screen, after you start speaking | **0.34 s** (Hindi pinned) / 0.73 s (auto-detect) |
| Line finalises, after you stop speaking | **0.58–0.61 s** |
| Words recognised | accurate on the first pass, no cleanup needed |

For comparison, the old `live.py` chunked-Whisper pipeline showed text 12–25 seconds
late. This is the difference between reading along and reading a transcript.

## What has been tested, and what hasn't

Tested on the live Sarvam service with a real key: connection and authentication, the
exact query parameters (the service echoes them back in `session.begin`, and they match),
partial and final events, `vad.speech_start` / `vad.speech_end` timing, all four Script
modes, Hindi-pinned vs auto-detect, `fast` vs `balanced`, and the stop path. Transcription
accuracy was correct on every run.

Tested here, automatically (`selftest.py`, 31 checks): the browser protocol, ordering, the
50 ms audio frames, every byte of audio reaching `audio.wav`, reconnect after a dropped
socket, a refused session producing a readable error, parameter validation, and the stop
path (the last utterance's final is delivered before the page is closed). The audio
worklet's resampling is unit-tested at 48 kHz, 44.1 kHz and 16 kHz, including that tones
above 8 kHz are filtered out instead of aliasing.

Not tested: your microphone, the browser's permission prompt, and your own voice. If the
key is right and `selftest.py` passes, those are the only unknowns — and each of them
shows a clear message on screen rather than failing silently.

## Knobs, if you want them

- `--port 8123` — different port.
- `--no-browser` — don't open the browser automatically.
- `--host 0.0.0.0` — let other devices on your network open the page. Note the audio
  then crosses the network unencrypted; keep it to localhost unless you need it.
- Model: `saaras:v3-realtime` is the default. `saaras:v4` is also available on this
  endpoint and accepts the same protocol; to try it, add `"model": "saaras:v4"` to the
  `start` message in `static/index.html` (search for `stream_type`) — or ask me and I'll
  add a dropdown.
