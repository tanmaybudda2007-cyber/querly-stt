"""
Connection from querly-live to the backend, which does the LLM work.

querly-live transcribes; every finished line is also sent to the backend
(backend/live_llm.py), which keeps the conversation's context and runs the LLM
(summaries, brainstorming, fact checks, questions). This module is the client side:

    - jobs (lines, updates, questions, the final output) run one at a time, in order,
      so the backend always has every line before an update that should include it;
    - results are pushed to the page and saved in the session folder
      (llm.jsonl: every LLM call with the exact prompt; output.md: the final output);
    - if the backend restarts mid-session it forgets the session, so a new one is
      started and the transcript so far is sent again;
    - if the backend can't be reached, transcription carries on without the LLM.

BACKEND_URL (optional, default http://127.0.0.1:8000) is where the backend runs.
Only the standard library is used, so querly-live needs no extra packages.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from datetime import datetime

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
LLM_TIMEOUT = 240  # seconds; the backend itself retries rate-limited LLM calls


def backend_url():
    return (os.environ.get("BACKEND_URL") or DEFAULT_BACKEND_URL).strip().rstrip("/")


class BackendError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status  # HTTP status, or None if the backend couldn't be reached
        self.message = message


def _request(method, path, body=None, timeout=LLM_TIMEOUT):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(backend_url() + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"null")
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", errors="replace")
        try:
            text = json.loads(text).get("detail", text)
        except (ValueError, AttributeError):
            pass
        raise BackendError(e.code, str(text)) from None
    except (urllib.error.URLError, OSError) as e:
        reason = getattr(e, "reason", e)
        raise BackendError(None, f"Can't reach the backend at {backend_url()} ({reason}). Start it "
                                 "with: cd backend, then uvicorn app:app --port 8000") from None


async def call(method, path, body=None, timeout=LLM_TIMEOUT):
    """Blocking HTTP in a worker thread, so the audio path never waits on it."""
    return await asyncio.to_thread(_request, method, path, body, timeout)


async def fetch_options():
    """LLM settings for the page, or why the LLM isn't available."""
    try:
        options = await call("GET", "/api/live/options", timeout=3)
        return {"available": True, "backend": backend_url(), **options}
    except BackendError as e:
        return {"available": False, "backend": backend_url(), "reason": e.message}


def clean_settings(raw):
    """The page's LLM settings -> the backend's /api/live/start body."""

    def number(key, cast, default=None):
        try:
            return cast(raw[key]) if raw.get(key) not in (None, "") else default
        except (TypeError, ValueError):
            return default

    def text(key):
        value = raw.get(key)
        return value.strip() if isinstance(value, str) and value.strip() else None

    return {
        "mode": text("mode") or "summarize",
        "prompt": text("prompt"),
        "provider": text("provider"),
        "model": text("model"),
        "temperature": number("temperature", float),
        "max_tokens": number("max_tokens", int),
        "update_every": max(0.0, number("update_every", float, 120.0)),
        "recent_minutes": max(0.5, number("recent_minutes", float, 4.0)),
    }


class LLMBridge:
    def __init__(self, settings, folder, send):
        self.settings = settings  # body for /api/live/start
        self.folder = folder  # session folder, for llm.jsonl and output.md
        self.send = send  # async fn(dict): message to the page
        self.session_id = None
        self.lines = []  # every line, to rebuild the session after a backend restart
        self.sent = 0  # how many of self.lines the backend has
        self.unreachable = False  # backend down: say so once, not for every line
        self.jobs = asyncio.Queue()
        self.worker = None
        self.final_written = False
        self.last_output = None

    # ------------------------------------------------------------ used by live_server.py

    async def start(self):
        try:
            self.session_id = (await call("POST", "/api/live/start", self.settings, 15))["session_id"]
        except BackendError as e:
            await self.send({"type": "llm_error", "fatal": True,
                             "message": f"LLM is off for this session: {e.message}"})
            return False
        self.worker = asyncio.create_task(self._work())
        every = self.settings["update_every"]
        await self.send({"type": "llm_status", "ready": True, "message":
                         f"LLM on: {self.settings['mode'] if not self.settings['prompt'] else 'custom prompt'}"
                         + (f", updates every {every:.0f} s of speech" if every else ", updates on request")})
        return True

    def add_line(self, text, start, end, language):
        if self.worker:
            self.jobs.put_nowait(("line", {"text": text, "start": start, "end": end,
                                           "language": language}))

    def request_update(self):
        if self.worker:
            self.jobs.put_nowait(("update", None))

    def ask(self, question):
        if self.worker and question:
            self.jobs.put_nowait(("question", question))

    async def finish(self, timeout=LLM_TIMEOUT):
        """Queue the final output and wait for everything queued before it."""
        if not self.worker:
            return
        self.jobs.put_nowait(("final", None))
        self.jobs.put_nowait(None)
        try:
            await asyncio.wait_for(self.worker, timeout)
        except asyncio.TimeoutError:
            self.worker.cancel()
            await self.send({"type": "llm_error", "message": "The final LLM output took too long."})
        if not self.final_written and self.last_output:
            # No final (e.g. it failed): keep the latest update as the session's output.
            (self.folder / "output.md").write_text(self.last_output + "\n", encoding="utf-8")

    # ------------------------------------------------------------ worker

    async def _work(self):
        while True:
            job = await self.jobs.get()
            if job is None:
                return
            kind, value = job
            step = kind
            try:
                if kind == "line":
                    self.lines.append(value)
                    if await self._send_lines():
                        step = "update"
                        await self._run("update")
                else:
                    await self._send_lines()  # anything missed while the backend was down
                    await self._run(kind, value)
                if self.unreachable:
                    self.unreachable = False
                    await self.send({"type": "llm_status", "message": "The backend is reachable "
                                     "again; the transcript is up to date there."})
            except Exception as e:  # never let the LLM break transcription
                if isinstance(e, BackendError) and e.status is None:
                    if not self.unreachable:
                        self.unreachable = True
                        await self.send({"type": "llm_error", "message": f"{e.message}. Lines are "
                                         "kept and sent when it's reachable again."})
                    continue
                message = e.message if isinstance(e, BackendError) else str(e)
                await self.send({"type": "llm_error", "message": f"LLM {step} failed: {message}"})

    async def _send_lines(self):
        """Send lines the backend doesn't have yet, in order. True if an update is due."""
        due = False
        while self.sent < len(self.lines):
            result = await self._session_call("transcript", self.lines[self.sent], timeout=15,
                                              replayed=True)
            if result is None:  # the session was rebuilt, which sent every line
                return False
            self.sent += 1
            due = due or bool(result.get("update_due"))
        return due

    async def _run(self, kind, value=None):
        await self.send({"type": "llm_working", "kind": kind})
        body = {"question": value} if kind == "question" else None
        result = await self._session_call(kind, body)
        if not result:
            await self.send({"type": "llm_status", "message": "Nothing new to update yet."
                             if kind == "update" else "Nothing was transcribed, so there is no output."})
            return
        self._save(result)
        if result["kind"] in ("update", "final"):
            self.last_output = result["output"]
        await self.send({"type": "llm", "kind": result["kind"], "output": result["output"],
                         "question": result.get("question"), "provider": result.get("provider"),
                         "model": result.get("model"), "seconds": result.get("seconds")})

    async def _session_call(self, action, body=None, timeout=LLM_TIMEOUT, replayed=False):
        """POST /api/live/<session>/<action>. If the backend no longer knows the session
        (it restarted), start a new one, resend the transcript and try again once.
        replayed=True means `body` is a line that the resend already included."""
        try:
            return await call("POST", f"/api/live/{self.session_id}/{action}", body, timeout)
        except BackendError as e:
            if e.status != 404:
                raise
        self.session_id = (await call("POST", "/api/live/start", self.settings, 15))["session_id"]
        for line in self.lines:
            await call("POST", f"/api/live/{self.session_id}/transcript", line, 15)
        self.sent = len(self.lines)
        await self.send({"type": "llm_status", "message": "The backend restarted, so the LLM "
                         "session was rebuilt from the transcript so far."})
        if replayed:
            return None
        return await call("POST", f"/api/live/{self.session_id}/{action}", body, timeout)

    def _save(self, result):
        record = {"time": datetime.now().isoformat(timespec="seconds"), **result}
        with open(self.folder / "llm.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if result["kind"] == "final":
            (self.folder / "output.md").write_text(result["output"] + "\n", encoding="utf-8")
            self.final_written = True
