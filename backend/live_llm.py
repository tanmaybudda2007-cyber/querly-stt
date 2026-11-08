"""
LLM side of live mode.

querly-live does the transcription. It sends each finished line here, with the line's
start and end time in the recording; this module keeps the conversation's context and
runs the LLM: periodic updates, questions and a final output, in the "mode" chosen
(summarize, brainstorm, verify, ...).

How the LLM sees a long conversation: resending the whole transcript on every update
would grow without limit and quickly exhaust free API tiers. Instead the LLM gets
    - a running summary of everything older than `recent_minutes` (kept up to date by
      extra "compress" calls, prompts/live_compress.txt), and
    - the recent transcript word for word, split into the part it has already seen
      and the part that is new since its previous update.

Prompt templates are prompts/live_<mode>.txt. Add a file to add a mode. Placeholders:
    {summary} {recent} {new} {transcript} (= recent + new) {previous_output}
    {question} {language} {status} {elapsed}

Sessions live in memory: if the backend restarts, querly-live starts a new session and
sends the transcript again.
"""

import threading
import time
import uuid
from pathlib import Path

from llm import PROVIDERS, chat, default_settings, is_available
from transcribers import LANGUAGES

PROMPTS_DIR = Path(__file__).with_name("prompts")
INTERNAL_PROMPTS = {"compress", "question"}  # used by every mode, not modes themselves
SESSION_IDLE_LIMIT = 6 * 3600  # forget sessions nobody has touched for 6 hours


def available_modes():
    """{mode name: prompt template} for every prompts/live_<mode>.txt."""
    modes = {}
    for path in sorted(PROMPTS_DIR.glob("live_*.txt")):
        name = path.stem[len("live_"):]
        if name not in INTERNAL_PROMPTS:
            modes[name] = path.read_text(encoding="utf-8")
    return modes


def clock(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def as_text(entries):
    return "\n".join(f"[{clock(e['start'])}] {e['text']}" for e in entries)


def fill(template, values):
    """Plain replace (not str.format) so other braces in a prompt are left alone."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


def language_name(code):
    """'hi-IN' or 'hi' -> 'Hindi'."""
    if not code:
        return "an unknown language"
    base = code.split("-")[0].lower()
    return LANGUAGES.get("or" if base == "od" else base, code)


class LiveSession:
    def __init__(self, mode="summarize", prompt=None, provider=None, model=None,
                 temperature=None, max_tokens=None, update_every=120.0, recent_minutes=4.0):
        modes = available_modes()
        if not prompt and mode not in modes:
            raise ValueError(f"Unknown mode '{mode}'. Available: {', '.join(modes)}")
        self.mode = mode
        self.template = prompt or modes[mode]  # a custom prompt overrides the mode's file
        self.question_template = (PROMPTS_DIR / "live_question.txt").read_text(encoding="utf-8")
        self.compress_template = (PROMPTS_DIR / "live_compress.txt").read_text(encoding="utf-8")
        self.llm = {"provider": provider, "model": model, "temperature": temperature,
                    "max_tokens": max_tokens}
        self.update_every = float(update_every or 0)  # seconds of speech; 0 = only on request
        self.window = float(recent_minutes) * 60

        self.entries = []  # finished lines: {"start", "end", "text", "language"}
        self.entries_lock = threading.Lock()  # short: adding and reading lines
        self.llm_lock = threading.Lock()  # long: one LLM operation at a time per session
        self.summary = ""  # running summary of entries[:summarized_upto]
        self.summarized_upto = 0
        self.updated_upto = 0  # entries the LLM has already seen in an update
        self.previous_output = ""
        self.update_pending = False
        self.language = None
        self.last_used = time.time()

    # ---------------------------------------------------------------- transcript

    def add(self, text, start, end, language=None):
        """Add a finished line. Returns True when an automatic update is due."""
        self.last_used = time.time()
        text = (text or "").strip()
        if not text:
            return False
        with self.entries_lock:
            self.entries.append({"start": float(start), "end": float(end), "text": text,
                                 "language": language})
            if language:
                self.language = language
            unseen = sum(e["end"] - e["start"] for e in self.entries[self.updated_upto:])
            due = bool(self.update_every) and not self.update_pending and unseen >= self.update_every
            if due:
                self.update_pending = True
        return due

    def snapshot(self):
        with self.entries_lock:
            return list(self.entries)

    # ---------------------------------------------------------------- LLM work

    def update(self, final=False):
        """Refresh the mode's output. Returns None if there is nothing new."""
        self.last_used = time.time()
        with self.llm_lock:
            try:
                entries = self.snapshot()
                if not entries or (not final and self.updated_upto >= len(entries)):
                    return None
                values = self._context(entries)
                values["status"] = ("The conversation has ended; write the final version." if final
                                    else "The conversation is still going on.")
                result = self._call(fill(self.template, values))
                self.previous_output = result["output"]
                self.updated_upto = len(entries)
                return {**result, "kind": "final" if final else "update"}
            finally:
                self.update_pending = False

    def ask(self, question):
        self.last_used = time.time()
        with self.llm_lock:
            entries = self.snapshot()
            if not entries:
                raise ValueError("Nothing has been transcribed yet.")
            values = self._context(entries)
            values["status"] = "The conversation is still going on."
            values["question"] = question
            result = self._call(fill(self.question_template, values))
            return {**result, "kind": "question", "question": question}

    def _context(self, entries):
        """Compress old parts if needed, then build the placeholder values."""
        self._compress(entries)
        seen_from = max(self.summarized_upto, min(self.updated_upto, len(entries)))
        recent, new = entries[self.summarized_upto:seen_from], entries[seen_from:]
        return {
            "summary": self.summary or "(nothing yet; the conversation has just started)",
            "recent": as_text(recent) or "(none)",
            "new": as_text(new) or "(nothing new)",
            "transcript": as_text(recent + new),
            "previous_output": self.previous_output or "(none yet)",
            "question": "",
            "language": language_name(self.language),
            "elapsed": clock(entries[-1]["end"]),
        }

    def _compress(self, entries):
        """Fold transcript older than the window into the running summary."""

        def too_long(i):  # is entries[i:] longer than the window?
            return entries[-1]["end"] - entries[i]["start"] > self.window

        while self.summarized_upto < len(entries) - 1 and too_long(self.summarized_upto):
            # Fold lines until the rest fits in the window, but at most one window's
            # worth per call, so each call stays small.
            first = self.summarized_upto
            k = first + 1
            while k < len(entries) - 1 and too_long(k) and \
                    entries[k]["end"] - entries[first]["start"] <= self.window:
                k += 1
            prompt = fill(self.compress_template, {
                "summary": self.summary or "(empty)",
                "transcript": as_text(entries[first:k]),
                "language": language_name(self.language),
            })
            self.summary = self._call(prompt)["output"]
            self.summarized_upto = k

    def _call(self, prompt):
        result = chat([{"role": "user", "content": prompt}], **self.llm)
        return {**result, "prompt": prompt}


# ------------------------------------------------------------------ session registry

_sessions = {}
_registry_lock = threading.Lock()


def options():
    """What the live page needs to show its LLM settings."""
    providers = {}
    for name, config in PROVIDERS.items():
        ok, note = is_available(name)
        providers[name] = {"available": ok, "note": note, "default_model": config["model"]}
    return {"modes": available_modes(), "providers": providers,
            "defaults": {"mode": "summarize", "provider": default_settings()["provider"],
                         "update_every": 120, "recent_minutes": 4}}


def start_session(**settings):
    provider = settings.get("provider") or default_settings()["provider"]
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider '{provider}'")
    ok, note = is_available(provider)
    if not ok:
        raise ValueError(f"LLM provider '{provider}' is not set up: {note}")
    settings["provider"] = provider
    session = LiveSession(**settings)
    session_id = uuid.uuid4().hex[:12]
    with _registry_lock:
        now = time.time()
        for old in [k for k, s in _sessions.items() if now - s.last_used > SESSION_IDLE_LIMIT]:
            del _sessions[old]
        _sessions[session_id] = session
    return session_id


def get_session(session_id):
    with _registry_lock:
        return _sessions.get(session_id)


def end_session(session_id):
    with _registry_lock:
        _sessions.pop(session_id, None)
