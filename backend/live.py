"""
Live transcription + LLM.

Records from the microphone (or plays a .wav file as if it were live), cuts the audio
into chunks of about 15 s at natural pauses, transcribes each chunk as soon as it is
cut, and every so often sends the conversation so far to an LLM to summarize it,
brainstorm on it, check its facts, or anything else a prompt file asks for.

    python live.py                                   # microphone, settings from .env
    python live.py --language hi --mode brainstorm
    python live.py --engine sarvam --provider gemini --update-every 60
    python live.py --input meeting.wav               # simulate live from a file
    python live.py --list-devices                    # find your microphone's number

While it runs, type in the terminal and press Enter:
    s             update the LLM output now
    ? <question>  ask the LLM something about the conversation so far
    q             stop (same as Ctrl+C)
When it stops, the last chunk is transcribed and the LLM writes a final version.

How it works (four stages in separate threads, connected by queues, so a slow
transcription or LLM call never makes the recording lose audio):

    source -> segmenter -> transcriber -> LLM worker
    mic or    cuts at a    one chunk at   periodic updates, questions,
    file      pause        a time         final output

How the LLM sees a long conversation: resending the whole transcript on every update
would grow without limit and quickly exhaust free API tiers. Instead the LLM gets
    - a running summary of everything older than --recent-minutes (kept up to date by
      extra "compress" calls, prompts/live_compress.txt), and
    - the recent transcript word for word, split into the part it has already seen
      and the part that is new since its previous update.

Prompt templates are prompts/live_<mode>.txt. Add a file to add a mode. Placeholders:
    {summary} {recent} {new} {transcript} (= recent + new) {previous_output}
    {question} {language} {status} {elapsed}

Everything is saved in sessions/<date_time>/:
    audio.wav, transcript.jsonl (one line per chunk), transcript.txt,
    llm.jsonl (every LLM call), output.md (final LLM output), settings.json
"""

import argparse
import json
import os
import queue
import sys
import threading
import time
from datetime import datetime
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from llm import PROVIDERS, chat, is_available
from transcribers import LANGUAGES, SAMPLE_RATE, get_transcriber, load_audio

PROMPTS_DIR = Path(__file__).with_name("prompts")
SPECIAL_PROMPTS = {"compress", "question"}  # used internally, not selectable as --mode

print_lock = threading.Lock()


def log(message=""):
    """Print from any thread without lines getting mixed up."""
    with print_lock:
        print(message, flush=True)


def clock(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def available_modes():
    return sorted(p.stem[len("live_"):] for p in PROMPTS_DIR.glob("live_*.txt")
                  if p.stem[len("live_"):] not in SPECIAL_PROMPTS)


# ---------------------------------------------------------------- 1. audio sources

def microphone_source(args, audio_q, stop):
    """Put 0.1 s blocks of 16 kHz mono audio from the microphone into audio_q."""
    try:
        import sounddevice as sd
    except (ImportError, OSError) as e:
        raise SystemExit(f"Microphone input needs sounddevice + PortAudio ({e}).\n"
                         "  pip install sounddevice\n"
                         "  Linux: sudo apt install libportaudio2\n"
                         "Or use --input file.wav to simulate from a file.") from None

    def open_stream(rate):
        g = gcd(int(rate), SAMPLE_RATE)

        def callback(indata, frames, time_info, status):
            if status:
                log(f"(audio warning: {status})")
            block = indata[:, 0].copy()
            if rate != SAMPLE_RATE:  # only if the device can't record at 16 kHz itself
                block = resample_poly(block, SAMPLE_RATE // g, int(rate) // g).astype(np.float32)
            audio_q.put(block)

        return sd.InputStream(samplerate=rate, channels=1, dtype="float32", device=args.device,
                              blocksize=int(rate) // 10, callback=callback)

    try:
        try:
            stream = open_stream(SAMPLE_RATE)
        except sd.PortAudioError:
            rate = sd.query_devices(args.device, "input")["default_samplerate"]
            log(f"(the microphone can't record at 16 kHz; recording at {rate:.0f} Hz and resampling)")
            stream = open_stream(rate)
    except (sd.PortAudioError, ValueError) as e:
        raise SystemExit(f"Could not open the microphone ({e}). Check that one is connected and "
                         "allowed, or pick one with --device (see --list-devices).") from None
    with stream:
        log("Listening... (type 's' + Enter to update, '? question' to ask, 'q' or Ctrl+C to stop)\n")
        stop.wait()


def file_source(args, audio_q, stop):
    """Feed a .wav file in 0.1 s blocks, at real-time speed times --speed (0 = no waiting)."""
    audio = load_audio(args.input)
    block = SAMPLE_RATE // 10
    log(f"Playing {args.input} ({clock(len(audio) / SAMPLE_RATE)}) as live input"
        + (f" at {args.speed}x speed" if args.speed else ", as fast as possible") + "\n")
    start = time.time()
    for i in range(0, len(audio), block):
        if stop.is_set():
            break
        audio_q.put(audio[i : i + block])
        if args.speed > 0:
            delay = start + (i + block) / SAMPLE_RATE / args.speed - time.time()
            if delay > 0:
                time.sleep(delay)


def run_source(args, audio_q, stop):
    try:
        (file_source if args.input else microphone_source)(args, audio_q, stop)
    except SystemExit as e:
        log(str(e))
    except Exception as e:
        log(f"Audio input error: {e}")
    finally:
        audio_q.put(None)  # tells the segmenter there's no more audio
        stop.set()


# ---------------------------------------------------------------- 2. segmenter

def quietest_point(audio, start, end, frame=480):
    """Middle of the lowest-energy 30 ms frame in audio[start:end]."""
    start = max(0, start)
    frames = audio[start:end][: (end - start) // frame * frame].reshape(-1, frame)
    return start + int(np.argmin((frames**2).mean(axis=1))) * frame + frame // 2


class Segmenter:
    """Collects audio and cuts it into chunks.

    --cut pause (default): once a chunk is at least --min-chunk seconds long, cut it at
    the next pause of --pause seconds; if nobody pauses, cut at --max-chunk. Cutting in
    a pause instead of every N seconds avoids splitting words, which would be
    misrecognized on both sides of the cut.
    --cut fixed: cut every --chunk-seconds regardless (for comparison).

    Chunks without speech are dropped instead of transcribed, because Whisper tends to
    invent text ("Thanks for watching") on silence. Speech is detected with Silero VAD,
    which comes with faster-whisper.
    """

    def __init__(self, args, audio_q, chunk_q, wav_path):
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        self.args, self.audio_q, self.chunk_q = args, audio_q, chunk_q
        self.find_speech = get_speech_timestamps
        self.vad_options = VadOptions(min_silence_duration_ms=int(args.pause * 1000), speech_pad_ms=100)
        self.wav = sf.SoundFile(wav_path, "w", samplerate=SAMPLE_RATE, channels=1,
                                subtype="PCM_16") if wav_path else None
        self.count = 0
        self.silent_chunks = 0

    def run(self):
        buffer = np.zeros(0, dtype=np.float32)
        offset = 0  # position of buffer[0] in the whole recording, in samples
        new_samples = 0
        try:
            while True:
                block = self.audio_q.get()
                if block is None:
                    break
                if self.wav:
                    self.wav.write(block)
                buffer = np.concatenate([buffer, block])
                new_samples += len(block)
                if new_samples < SAMPLE_RATE // 2:  # look for a cut twice per second
                    continue
                new_samples = 0
                cut = self.find_cut(buffer)
                if cut:
                    self.emit(buffer[:cut], offset)
                    buffer, offset = buffer[cut:], offset + cut
            if len(buffer) > SAMPLE_RATE // 4:  # whatever is left when recording stops
                self.emit(buffer, offset)
        finally:
            if self.wav:
                self.wav.close()
            self.chunk_q.put(None)

    def find_cut(self, buffer):
        """Return where to cut the buffer (in samples), or None to keep collecting."""
        n = len(buffer)
        if self.args.cut == "fixed":
            size = int(self.args.chunk_seconds * SAMPLE_RATE)
            return size if n >= size else None

        min_len = int(self.args.min_chunk * SAMPLE_RATE)
        max_len = int(self.args.max_chunk * SAMPLE_RATE)
        pause_len = int(self.args.pause * SAMPLE_RATE)
        if n < min_len:
            return None

        speech = self.find_speech(buffer, self.vad_options)
        if not speech:
            return n - pause_len  # only silence: drop it, keeping the end in case speech is starting
        # Pauses: gaps between detected speech segments (Silero already merges gaps
        # shorter than --pause), plus a long enough silence at the end of the buffer.
        pauses = [(a["end"], b["start"]) for a, b in zip(speech, speech[1:])]
        last_end = speech[-1]["end"]
        if n - last_end >= pause_len:
            pauses.append((last_end, n))
        for start, end in pauses:
            middle = (start + end) // 2
            if middle >= min_len:
                return middle
        if n >= max_len:
            # Nobody paused long enough: cut in the longest pause we have, if it isn't
            # too early, otherwise at the quietest moment of the last 3 seconds (usually
            # a gap between words).
            usable = [(e - s, (s + e) // 2) for s, e in pauses if (s + e) // 2 >= min_len // 2]
            return max(usable)[1] if usable else quietest_point(buffer, max_len - 3 * SAMPLE_RATE, max_len)
        return None

    def emit(self, audio, offset):
        if not self.find_speech(audio, self.vad_options):
            self.silent_chunks += 1
            return
        self.count += 1
        self.chunk_q.put({"index": self.count, "start": offset / SAMPLE_RATE,
                          "end": (offset + len(audio)) / SAMPLE_RATE, "audio": audio})


# ---------------------------------------------------------------- 3. transcriber

class Session:
    """Everything known about the conversation so far, shared by the threads."""

    def __init__(self, folder):
        self.folder = folder
        self.lock = threading.Lock()
        self.entries = []  # one per transcribed chunk: index, start, end, text, language
        self.language = None
        self.summary = ""  # running summary of entries[:summarized_upto]
        self.summarized_upto = 0
        self.updated_upto = 0  # entries the LLM has already seen in an update
        self.previous_output = ""
        self.update_pending = False

    def add(self, entry):
        with self.lock:
            self.entries.append(entry)
        with open(self.folder / "transcript.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def snapshot(self):
        with self.lock:
            return list(self.entries)

    def unseen_seconds(self):
        with self.lock:
            return sum(e["end"] - e["start"] for e in self.entries[self.updated_upto:])

    def elapsed(self):
        with self.lock:
            return self.entries[-1]["end"] if self.entries else 0


def run_transcriber(args, session, transcriber, chunk_q, llm_q):
    language = args.language
    previous_text = ""
    last_warning = 0
    while True:
        chunk = chunk_q.get()
        if chunk is None:
            break
        # The end of the previous transcript, so names and spelling stay consistent.
        prompt = previous_text[-args.context_chars:] if args.context_chars else None
        started = time.time()
        try:
            result = transcriber.transcribe_audio(chunk["audio"], language, args.task, prompt)
        except Exception as e:
            log(f"[{clock(chunk['start'])}] transcription error: {e}")
            continue
        took = time.time() - started
        text = result["text"].strip()
        if not text:
            continue

        if language is None and not args.redetect:
            language = result["language"]  # detect once, then keep it: per-chunk detection is noisy
            log(f"(language: {LANGUAGES.get(language, language)}; fixed from now on, "
                "use --language or --redetect to change this)")
        session.language = result["language"]
        entry = {"index": chunk["index"], "start": round(chunk["start"], 2),
                 "end": round(chunk["end"], 2), "text": text, "language": result["language"],
                 "transcribe_seconds": round(took, 2)}
        session.add(entry)
        log(f"[{clock(chunk['start'])}] {text}")
        previous_text = f"{previous_text} {text}"[-1000:]

        duration = chunk["end"] - chunk["start"]
        waiting = chunk_q.qsize()
        if (took > duration or waiting >= 2) and time.time() - last_warning > 60:
            last_warning = time.time()
            log(f"(transcription is slower than real time: {took:.1f} s for {duration:.1f} s of "
                f"audio, {waiting} chunk(s) waiting; try a smaller --whisper-model or --engine sarvam)")

        if (llm_q and args.update_every and not session.update_pending
                and session.unseen_seconds() >= args.update_every):
            session.update_pending = True
            llm_q.put(("update", None))


# ---------------------------------------------------------------- 4. LLM worker

def as_text(entries):
    return "\n".join(f"[{clock(e['start'])}] {e['text']}" for e in entries)


def fill(template, values):
    """Plain replace (not str.format) so other braces in a prompt are left alone."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


class LLMWorker:
    def __init__(self, args, session, llm_q):
        self.args, self.session, self.llm_q = args, session, llm_q
        prompt_file = Path(args.prompt) if args.prompt else PROMPTS_DIR / f"live_{args.mode}.txt"
        self.mode_prompt = prompt_file.read_text(encoding="utf-8")
        self.question_prompt = (PROMPTS_DIR / "live_question.txt").read_text(encoding="utf-8")
        self.compress_prompt = (PROMPTS_DIR / "live_compress.txt").read_text(encoding="utf-8")

    def run(self):
        while True:
            job = self.llm_q.get()
            if job is None:
                break
            kind, question = job
            try:
                self.handle(kind, question)
            except Exception as e:  # never stop transcription because of the LLM
                log(f"(LLM {kind} failed: {e}" + ("; will try again at the next update)"
                                                    if kind == "update" else ")"))
            finally:
                if kind == "update":
                    self.session.update_pending = False

    def call(self, kind, prompt):
        a = self.args
        result = chat([{"role": "user", "content": prompt}], a.provider, a.model,
                      a.temperature, a.max_tokens)
        with open(self.session.folder / "llm.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "kind": kind,
                                "elapsed": round(self.session.elapsed(), 1), "prompt": prompt,
                                **result}, ensure_ascii=False) + "\n")
        return result

    def compress(self, entries):
        """Fold transcript older than --recent-minutes into the running summary."""
        s = self.session
        window = self.args.recent_minutes * 60

        def too_long(i):  # is entries[i:] longer than the window?
            return entries[-1]["end"] - entries[i]["start"] > window

        while s.summarized_upto < len(entries) - 1 and too_long(s.summarized_upto):
            # Fold entries until the rest fits in the window, but at most one window's
            # worth per call, so each call stays small.
            first = s.summarized_upto
            k = first + 1
            while k < len(entries) - 1 and too_long(k) and \
                    entries[k]["end"] - entries[first]["start"] <= window:
                k += 1
            part = entries[first:k]
            prompt = fill(self.compress_prompt, {"summary": s.summary or "(empty)",
                                                 "transcript": as_text(part),
                                                 "language": self.language_name()})
            s.summary = self.call("compress", prompt)["output"]
            s.summarized_upto = k

    def language_name(self):
        lang = self.session.language
        return LANGUAGES.get(lang, lang or "an unknown language")

    def handle(self, kind, question):
        s = self.session
        entries = s.snapshot()
        if not entries:
            if kind != "update":
                log("(nothing has been transcribed yet)")
            return
        if kind == "update" and s.updated_upto >= len(entries):
            return  # nothing new
        self.compress(entries)

        seen_from = max(s.summarized_upto, min(s.updated_upto, len(entries)))
        recent, new = entries[s.summarized_upto:seen_from], entries[seen_from:]
        ended = kind == "final"
        values = {
            "summary": s.summary or "(nothing yet; the conversation has just started)",
            "recent": as_text(recent) or "(none)",
            "new": as_text(new) or "(nothing new)",
            "transcript": as_text(recent + new),
            "previous_output": s.previous_output or "(none yet)",
            "question": question or "",
            "language": self.language_name(),
            "status": "The conversation has ended; write the final version." if ended
                      else "The conversation is still going on.",
            "elapsed": clock(entries[-1]["end"]),
        }
        template = self.question_prompt if kind == "question" else self.mode_prompt
        result = self.call(kind, fill(template, values))

        if kind != "question":
            s.previous_output = result["output"]
            s.updated_upto = len(entries)
        title = {"update": f"LLM update ({self.args.prompt or self.args.mode})",
                 "question": f"Answer: {question}", "final": "Final LLM output"}[kind]
        log(f"\n──── {title} · {result['provider']} {result['model']} · {result['seconds']} s ────\n"
            f"{result['output']}\n{'─' * 60}\n")


def read_commands(session, llm_q, stop):
    """Keyboard commands typed in the terminal while recording."""
    for line in sys.stdin:
        line = line.strip()
        if line.lower() in ("q", "quit", "stop"):
            stop.set()
            return
        if not llm_q:
            log("(the LLM is off: started with --no-llm)")
        elif line.lower() in ("s", "update"):
            llm_q.put(("update", None))
        elif line.startswith("?") and line[1:].strip():
            llm_q.put(("question", line[1:].strip()))
        elif line:
            log("(commands: s = update now, ? question = ask, q = stop)")


# ---------------------------------------------------------------- main

def parse_args():
    modes = available_modes()
    p = argparse.ArgumentParser(description="Live transcription + LLM",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("input")
    g.add_argument("--input", help="simulate live input from this .wav file instead of the microphone")
    g.add_argument("--speed", type=float, default=1.0, help="with --input: playback speed (0 = as fast as possible)")
    g.add_argument("--device", type=int, default=None, help="microphone number (see --list-devices)")
    g.add_argument("--list-devices", action="store_true", help="list audio devices and exit")

    g = p.add_argument_group("transcription")
    g.add_argument("--engine", default="whisper", choices=["whisper", "sarvam"])
    g.add_argument("--whisper-model", help="overrides WHISPER_MODEL, e.g. small, medium, large-v3")
    g.add_argument("--sarvam-model", help="overrides SARVAM_MODEL")
    g.add_argument("--language", help="language spoken, e.g. hi (default: detect from the first chunk)")
    g.add_argument("--redetect", action="store_true", help="detect the language in every chunk")
    g.add_argument("--task", default="transcribe", choices=["transcribe", "translate"],
                   help="translate = output English text")
    g.add_argument("--context-chars", type=int, default=200,
                   help="characters of previous transcript given to Whisper for continuity (0 = off)")

    g = p.add_argument_group("chunking")
    g.add_argument("--cut", default="pause", choices=["pause", "fixed"],
                   help="pause = cut at a pause between --min-chunk and --max-chunk; fixed = every --chunk-seconds")
    g.add_argument("--min-chunk", type=float, default=12.0, help="seconds")
    g.add_argument("--max-chunk", type=float, default=25.0, help="seconds")
    g.add_argument("--pause", type=float, default=0.5, help="silence (seconds) that counts as a pause")
    g.add_argument("--chunk-seconds", type=float, default=15.0, help="chunk length for --cut fixed")

    g = p.add_argument_group("LLM")
    g.add_argument("--no-llm", action="store_true", help="only transcribe")
    g.add_argument("--mode", default="summarize", choices=modes,
                   help="which prompts/live_<mode>.txt to use")
    g.add_argument("--prompt", help="custom prompt template file (overrides --mode)")
    g.add_argument("--provider", choices=list(PROVIDERS), help="default: LLM_PROVIDER in .env")
    g.add_argument("--model", help="default: LLM_MODEL in .env, or the provider's default")
    g.add_argument("--temperature", type=float, help="default: LLM_TEMPERATURE in .env")
    g.add_argument("--max-tokens", type=int, help="default: LLM_MAX_TOKENS in .env")
    g.add_argument("--update-every", type=float, default=120.0,
                   help="seconds of new speech between automatic LLM updates (0 = only on request and at the end)")
    g.add_argument("--recent-minutes", type=float, default=4.0,
                   help="how much recent transcript the LLM sees word for word; older parts are summarized")
    g.add_argument("--no-final", action="store_true", help="skip the final LLM output when stopping")

    g = p.add_argument_group("output")
    g.add_argument("--out-dir", default="sessions", help="sessions are saved in a new folder here")
    g.add_argument("--no-audio", action="store_true", help="don't save the recording")
    args = p.parse_args()
    if args.cut == "pause" and args.min_chunk >= args.max_chunk:
        p.error("--min-chunk must be smaller than --max-chunk")
    return args


def main():
    args = parse_args()
    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return
    if args.whisper_model:
        os.environ["WHISPER_MODEL"] = args.whisper_model
    if args.sarvam_model:
        os.environ["SARVAM_MODEL"] = args.sarvam_model

    if not args.input:  # check the microphone library before loading models
        try:
            import sounddevice  # noqa: F401
        except (ImportError, OSError) as e:
            raise SystemExit(f"Microphone input needs sounddevice + PortAudio ({e}).\n"
                             "  pip install sounddevice\n"
                             "  Linux: sudo apt install libportaudio2\n"
                             "Or use --input file.wav to simulate from a file.") from None
    if not args.no_llm:
        provider = args.provider or os.getenv("LLM_PROVIDER", "groq")
        ok, reason = is_available(provider)
        if not ok:
            raise SystemExit(f"LLM provider '{provider}' is not set up: {reason}. "
                             "Choose another with --provider, or use --no-llm to only transcribe.")
    try:
        transcriber = get_transcriber(args.engine)  # load the model before recording starts
    except Exception as e:
        raise SystemExit(f"Could not start the {args.engine} engine: {e}") from None

    folder = Path(args.out_dir) / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder.mkdir(parents=True)
    (folder / "settings.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    session = Session(folder)
    audio_q, chunk_q = queue.Queue(), queue.Queue()
    llm_q = None if args.no_llm else queue.Queue()
    stop = threading.Event()

    segmenter = Segmenter(args, audio_q, chunk_q, None if args.no_audio else folder / "audio.wav")
    threads = {
        "source": threading.Thread(target=run_source, args=(args, audio_q, stop), daemon=True),
        "segmenter": threading.Thread(target=segmenter.run, daemon=True),
        "transcriber": threading.Thread(target=run_transcriber,
                                        args=(args, session, transcriber, chunk_q, llm_q), daemon=True),
    }
    if llm_q:
        worker = LLMWorker(args, session, llm_q)
        threads["llm"] = threading.Thread(target=worker.run, daemon=True)
    threading.Thread(target=read_commands, args=(session, llm_q, stop), daemon=True).start()

    llm_info = "off" if args.no_llm else (
        f"{args.provider or os.getenv('LLM_PROVIDER', 'groq')}, mode: {args.prompt or args.mode}, "
        + (f"update every {args.update_every:.0f} s of speech" if args.update_every else "updates on request"))
    log(f"Session: {folder}\nEngine: {args.engine} ({transcriber.model_name}) · "
        f"language: {args.language or 'auto'} · LLM: {llm_info}")
    for t in threads.values():
        t.start()

    try:
        while threads["source"].is_alive():
            threads["source"].join(0.2)
    except KeyboardInterrupt:
        stop.set()
    try:
        log("\nStopping: transcribing what's left...")
        threads["segmenter"].join()
        threads["transcriber"].join()
        if llm_q:
            if not args.no_final:
                log("Writing the final LLM output...")
                llm_q.put(("final", None))
            llm_q.put(None)
            threads["llm"].join()
    except KeyboardInterrupt:
        log("Stopped without finishing.")

    entries = session.snapshot()
    (folder / "transcript.txt").write_text(as_text(entries) + "\n", encoding="utf-8")
    if session.previous_output:
        (folder / "output.md").write_text(session.previous_output + "\n", encoding="utf-8")
    log(f"\n{len(entries)} chunks transcribed ({clock(session.elapsed())}), "
        f"{segmenter.silent_chunks} silent chunk(s) skipped. Saved in {folder}/")


if __name__ == "__main__":
    main()
