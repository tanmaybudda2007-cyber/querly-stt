#!/usr/bin/env python3
"""
Querly Live - press record, talk, watch your words appear.

The browser captures the microphone (so nothing native has to be installed:
no PortAudio, no sounddevice, no ffmpeg), sends 16 kHz mono PCM to this server
over a WebSocket, and this server streams that audio to Sarvam's realtime
speech-to-text socket. Partials and finals come back and are pushed straight
to the page, which renders them like song lyrics.

    python app.py                 # then open http://127.0.0.1:8000
    python app.py --no-browser    # don't open the browser automatically
    python app.py --port 8123

Environment (or a .env file next to this script):
    SARVAM_API_KEY      required - https://indus.sarvam.ai/key-management
    SARVAM_WS_URL       optional - defaults to the real Sarvam endpoint.
                        Point it at mock_sarvam.py to test without a key.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import wave
import webbrowser
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

import uvicorn
import websockets
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
SESSIONS_DIR = ROOT / "sessions"
DEFAULT_WS_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"

SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2  # linear16, mono

# Everything the browser is allowed to ask for. Anything else is rejected, so a
# typo in the page can never produce a confusing server-side error.
LANGUAGES = {
    "auto", "en-IN", "hi-IN", "bn-IN", "kn-IN", "ml-IN", "mr-IN", "or-IN", "pa-IN",
    "ta-IN", "te-IN", "gu-IN", "ur-IN", "ne-IN", "kok-IN", "ks-IN", "sd-IN", "sa-IN",
    "sat-IN", "mni-IN", "brx-IN", "mai-IN", "doi-IN", "as-IN",
}
MODES = {"transcribe", "translit", "codemix", "verbatim", "translate"}
STREAM_TYPES = {"fast", "balanced", "simulated"}

CLOSE_CODES = {
    1003: "Sarvam refused the session (rate limit, quota exhausted, or invalid API key).",
    1008: "Sarvam closed the session after a period of inactivity.",
    1011: "Sarvam hit an internal error. Try again in a moment.",
    4000: "Sarvam rejected a connection parameter (model, language or account).",
}


def log(message=""):
    """Print safely on a Windows console, which is not always UTF-8."""
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        print(message.encode("ascii", "replace").decode("ascii"), flush=True)


def load_dotenv(path):
    """Minimal .env reader: KEY=VALUE lines, # comments. Existing env wins."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv(ROOT / ".env")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def api_key():
    return (os.environ.get("SARVAM_API_KEY") or "").strip()


def ws_url():
    return (os.environ.get("SARVAM_WS_URL") or DEFAULT_WS_URL).strip()


# --------------------------------------------------------------------- sessions

class SessionSink:
    """Writes the recording and the transcript to sessions/<timestamp>/."""

    def __init__(self, base_dir):
        self.folder = Path(base_dir) / datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.folder.mkdir(parents=True, exist_ok=True)
        self._wav = wave.open(str(self.folder / "audio.wav"), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(SAMPLE_RATE)
        self._jsonl = open(self.folder / "transcript.jsonl", "a", encoding="utf-8")
        self._texts = []
        self._closed = False
        self.audio_bytes = 0

    def write_audio(self, pcm):
        if self._closed:
            return
        self._wav.writeframes(pcm)
        self.audio_bytes += len(pcm)

    def add_final(self, idx, text, language=None, generation=None):
        if self._closed or not text:
            return
        entry = {
            "session_gen": generation,
            "utterance_idx": idx,
            "time": datetime.now().isoformat(timespec="seconds"),
            "text": text,
            "language": language,
        }
        self._jsonl.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._jsonl.flush()
        self._texts.append(text)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._wav.close()
            self._jsonl.close()
            (self.folder / "transcript.txt").write_text(
                "\n".join(self._texts) + ("\n" if self._texts else ""), encoding="utf-8")
        except Exception as exc:  # never let cleanup hide the real error
            log(f"(could not finish writing the session: {exc})")


# ------------------------------------------------------------------ sarvam link

def build_url(cfg):
    query = {
        "language_code": cfg["language_code"],
        "model": cfg["model"],
        "stream_type": cfg["stream_type"],
        "mode": cfg["mode"],
        "endpointing": "vad",
        "encoding": "linear16",
        "sample_rate": str(SAMPLE_RATE),
        "threshold": str(cfg["threshold"]),
        "silence_duration_ms": str(cfg["silence_ms"]),
        "min_speech_duration_ms": "250",
    }
    return f"{ws_url()}?{urlencode(query)}"


async def open_socket(url, key):
    """websockets renamed extra_headers to additional_headers in v13; accept both."""
    kwargs = {"max_size": 2 ** 22, "ping_interval": 20, "ping_timeout": 20}
    try:
        return await websockets.connect(
            url, additional_headers={"api-subscription-key": key}, **kwargs)
    except TypeError:
        return await websockets.connect(
            url, extra_headers={"api-subscription-key": key}, **kwargs)


def describe_connect_error(exc):
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if status in (401, 403):
        return (f"Sarvam rejected the API key (HTTP {status}). "
                "Check SARVAM_API_KEY in .env - a new key is at "
                "https://indus.sarvam.ai/key-management")
    if status is not None:
        return f"Sarvam refused the connection (HTTP {status}): {exc}"
    return f"Could not reach Sarvam at {ws_url()} - {exc}"


def map_event(event):
    """Sarvam event -> message for the browser. None means 'ignore this one'."""
    kind = event.get("event")
    if kind == "transcript.partial":
        return {"type": "partial", "idx": event.get("utterance_idx"),
                "text": event.get("text", ""), "language": event.get("language")}
    if kind == "transcript.final":
        return {"type": "final", "idx": event.get("utterance_idx"),
                "text": event.get("text", ""), "language": event.get("language"),
                "language_confidence": event.get("language_confidence")}
    if kind in ("vad.speech_start", "vad.speech_end"):
        return {"type": "vad", "state": kind.split(".", 1)[1],
                "idx": event.get("utterance_idx")}
    if kind == "session.begin":
        return {"type": "ready", "request_id": event.get("request_id")}
    if kind == "session.end":
        return {"type": "ended", "audio_duration_s": event.get("audio_duration_s")}
    if kind == "error":
        return {"type": "error", "code": event.get("code"),
                "message": event.get("message", "unknown error"),
                "fatal": bool(event.get("is_fatal"))}
    return None


class SarvamLink:
    """One Sarvam socket, with a small amount of reconnect handling.

    If the socket dies mid-session (network blip, 1011), a new one is opened and
    the transcript simply continues; the browser is told what happened. Audio
    sent while the socket is down is dropped rather than queued forever.
    """

    MAX_RECONNECTS = 3

    def __init__(self, cfg, key, on_message):
        self.cfg = cfg
        self.key = key
        self.on_message = on_message      # async fn(dict) -> None
        self.ws = None
        self.reconnects = 0
        self.stopping = False
        self.ended = asyncio.Event()
        self._ready = asyncio.Event()
        self._send_lock = asyncio.Lock()
        self._tasks = []
        self._generation = 0

    async def connect(self):
        url = build_url(self.cfg)
        self.ws = await open_socket(url, self.key)
        self._generation += 1
        self._ready.set()
        self._tasks.append(asyncio.create_task(self._receive(self._generation, self.ws)))

    async def _receive(self, generation, socket):
        close_code = None
        try:
            async for raw in socket:
                try:
                    event = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                message = map_event(event)
                if message is not None:
                    # After a reconnect Sarvam starts a fresh session and its
                    # utterance_idx restarts at 0, so the page needs to know
                    # which session a line belongs to.
                    if message["type"] in ("partial", "final", "vad"):
                        message["gen"] = generation
                    if message["type"] == "error" and message.get("fatal"):
                        self.stopping = True  # nothing to retry against
                    await self.on_message(message)
                if event.get("event") == "session.end":
                    self.ended.set()
        except Exception as exc:
            close_code = getattr(exc, "code", None)
        finally:
            if generation == self._generation and not self.stopping:
                await self._recover(socket, close_code)

    async def _recover(self, socket, close_code=None):
        self._ready.clear()
        if self.ws is socket:
            self.ws = None
        if close_code in (1003, 4000):
            # A bad key, an exhausted quota or a rejected parameter will not fix
            # itself: say so instead of retrying three times.
            self.stopping = True
            await self.on_message({"type": "error", "fatal": True,
                                   "message": CLOSE_CODES[close_code]})
            return
        if self.reconnects >= self.MAX_RECONNECTS:
            await self.on_message({"type": "lost",
                                   "message": "Lost the connection to Sarvam and could not "
                                              "get it back. Recording stopped."})
            return
        self.reconnects += 1
        await self.on_message({"type": "status",
                               "message": "Connection dropped - reconnecting..."})
        for attempt in range(3):
            if self.stopping:
                return
            try:
                await self.connect()
                await self.on_message({"type": "status",
                                       "message": "Reconnected. Transcript continues."})
                return
            except Exception:
                await asyncio.sleep(0.5 * (attempt + 1))
        await self.on_message({"type": "lost",
                               "message": "Could not reconnect to Sarvam. Recording stopped."})

    async def send_audio(self, pcm):
        if self.stopping:
            return
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=5)
        except asyncio.TimeoutError:
            return  # socket is down; drop this frame rather than block the mic
        payload = json.dumps({"event": "audio_input",
                              "audio": base64.b64encode(pcm).decode("ascii")})
        async with self._send_lock:
            socket = self.ws
            if socket is None:
                return
            try:
                await socket.send(payload)
            except Exception:
                self.ws = None
                self._ready.clear()

    async def finish(self, wait_seconds=3.0):
        """Ask Sarvam to finalise whatever is buffered, then close."""
        self.stopping = True
        socket = self.ws
        if socket is not None:
            try:
                await socket.send(json.dumps({"event": "end"}))
            except Exception:
                pass
            try:
                await asyncio.wait_for(self.ended.wait(), timeout=wait_seconds)
            except asyncio.TimeoutError:
                pass
        # The last utterance's final often arrives together with session.end, so
        # let the receive loop hand over anything still in flight before the
        # browser socket is closed. Without this the final line can be written to
        # disk but never reach the screen.
        for task in list(self._tasks):
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except Exception:
                pass
        for task in self._tasks:
            task.cancel()
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None


# ------------------------------------------------------------------ web app

app = FastAPI(title="Querly Live")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
async def config():
    return JSONResponse({
        "has_key": bool(api_key()),
        "endpoint": ws_url(),
        "sample_rate": SAMPLE_RATE,
        "languages": sorted(LANGUAGES),
        "modes": sorted(MODES),
        "stream_types": sorted(STREAM_TYPES),
    })


@app.get("/healthz")
async def healthz():
    return {"ok": True}


def client_config(request):
    """Validate the browser's settings; fall back to good defaults."""
    cfg = {
        "language_code": request.get("language_code", "auto"),
        "mode": request.get("mode", "codemix"),
        "stream_type": request.get("stream_type", "fast"),
        "model": request.get("model", "saaras:v3-realtime"),
        "threshold": 0.3,
        "silence_ms": 400,
    }
    if cfg["language_code"] not in LANGUAGES:
        cfg["language_code"] = "auto"
    if cfg["mode"] not in MODES:
        cfg["mode"] = "codemix"
    if cfg["stream_type"] not in STREAM_TYPES:
        cfg["stream_type"] = "fast"
    if cfg["model"] not in ("saaras:v3-realtime", "saaras:v4"):
        cfg["model"] = "saaras:v3-realtime"
    try:
        silence = int(request.get("silence_ms", 400))
    except (TypeError, ValueError):
        silence = 400
    cfg["silence_ms"] = max(200, min(2000, silence))
    return cfg


@app.websocket("/ws")
async def stream(ws: WebSocket):
    await ws.accept()
    send_lock = asyncio.Lock()
    state = {"open": True}

    async def to_client(message):
        if not state["open"]:
            return
        async with send_lock:
            try:
                await ws.send_json(message)
            except Exception:
                state["open"] = False

    sink = None
    link = None
    try:
        try:
            first = await asyncio.wait_for(ws.receive_json(), timeout=20)
        except Exception:
            await to_client({"type": "error", "message": "No start message from the page."})
            return
        if not isinstance(first, dict) or first.get("type") != "start":
            await to_client({"type": "error", "message": "Expected a start message."})
            return
        if not api_key():
            await to_client({
                "type": "error",
                "fatal": True,
                "message": "SARVAM_API_KEY is not set. Put it in .env next to app.py "
                           "(see RUN_WINDOWS.md), then restart the server.",
            })
            return

        cfg = client_config(first)
        sink = SessionSink(SESSIONS_DIR)

        async def on_sarvam_message(message):
            # Everything goes to the page; finals are also written to disk, so a
            # session can be read back or re-transcribed with a bigger model later.
            if message.get("type") == "final" and sink is not None:
                sink.add_final(message.get("idx"), message.get("text"),
                               message.get("language"), message.get("gen"))
            await to_client(message)

        link = SarvamLink(cfg, api_key(), on_sarvam_message)
        try:
            await link.connect()
        except Exception as exc:
            await to_client({"type": "error", "fatal": True,
                             "message": describe_connect_error(exc)})
            return

        await to_client({"type": "started", "session": sink.folder.name,
                         "language_code": cfg["language_code"],
                         "mode": cfg["mode"], "stream_type": cfg["stream_type"]})
        log(f"session {sink.folder.name} started "
            f"({cfg['language_code']}, {cfg['mode']}, {cfg['stream_type']})")

        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes"):
                pcm = message["bytes"]
                sink.write_audio(pcm)
                await link.send_audio(pcm)
            elif message.get("text"):
                try:
                    control = json.loads(message["text"])
                except ValueError:
                    continue
                if control.get("type") == "stop":
                    break
    except Exception as exc:
        log(f"(stream error: {exc})")
        await to_client({"type": "error", "message": f"Unexpected error: {exc}"})
    finally:
        if link is not None:
            await link.finish()
        if sink is not None:
            sink.close()
            await to_client({"type": "closed", "session": sink.folder.name,
                             "seconds": round(sink.audio_bytes / BYTES_PER_SECOND, 1)})
            log(f"session {sink.folder.name} saved "
                f"({sink.audio_bytes / BYTES_PER_SECOND:.1f} s of audio)")
        state["open"] = False
        try:
            await ws.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Querly Live - realtime captions")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    log("")
    log("  Querly Live - realtime captions")
    log(f"  page:     {url}")
    log(f"  sarvam:   {ws_url()}")
    if not api_key():
        log("  API key:  MISSING - the page will show an error until you set "
            "SARVAM_API_KEY in .env")
    else:
        log(f"  API key:  set ({len(api_key())} characters)")
    log("")
    log("  Press Ctrl+C to stop the server.")
    log("")

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
