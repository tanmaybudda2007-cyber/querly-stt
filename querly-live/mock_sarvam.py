#!/usr/bin/env python3
"""
A fake Sarvam realtime socket, for testing the app without an API key.

IT DOES NOT TRANSCRIBE ANYTHING. It ignores the audio content and replays a
fixed list of Hinglish lines on a timer, but it speaks the same wire protocol
as Sarvam's realtime endpoint (session.begin, vad.*, transcript.partial,
transcript.final, session.end, error), and it rejects bad connection
parameters exactly like the real service does. That is enough to prove the
page, the server and the audio path all work end to end.

    python mock_sarvam.py                     # wss://... replaced by ws://127.0.0.1:8765
    python mock_sarvam.py --drop-after 4      # kill the socket after 4 s of audio
    python mock_sarvam.py --fail 1003         # refuse the session with a close code

Then run the app against it (PowerShell):

    $env:SARVAM_WS_URL="ws://127.0.0.1:8765/speech-to-text-realtime/ws"
    $env:SARVAM_API_KEY="dummy"
    python live_server.py
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from urllib.parse import parse_qs, urlparse

import websockets

LINES = [
    "so basically we need to decide if we ship the realtime transcription this week",
    "haan but pehle latency check karna padega, Sarvam ka socket kaisa perform karta hai",
    "I tested it, partial transcripts aa rahe hain under one second",
    "okay that's good, phir hum Wednesday ko demo kar sakte hain investor ko",
    "chalo phir main aaj hi deploy kar deta hoon",
]

VALID_LANGUAGES = {
    "auto", "en-IN", "hi-IN", "bn-IN", "kn-IN", "ml-IN", "mr-IN", "or-IN", "pa-IN",
    "ta-IN", "te-IN", "gu-IN", "ur-IN", "ne-IN", "kok-IN", "ks-IN", "sd-IN", "sa-IN",
    "sat-IN", "mni-IN", "brx-IN", "mai-IN", "doi-IN", "as-IN",
}
VALID_MODES = {"transcribe", "translit", "codemix", "verbatim", "translate"}
VALID_MODELS = {"saaras:v3-realtime", "saaras:v4"}

UTTERANCE_SECONDS = 3.0
PARTIAL_EVERY = 0.5


def log(message):
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        print(message.encode("ascii", "replace").decode("ascii"), flush=True)


def request_path(ws):
    path = getattr(ws, "path", None)
    if path:
        return path
    request = getattr(ws, "request", None)
    return getattr(request, "path", "") if request is not None else ""


def request_headers(ws):
    headers = getattr(ws, "request_headers", None)
    if headers is not None:
        return headers
    request = getattr(ws, "request", None)
    return getattr(request, "headers", {}) if request is not None else {}


def header(headers, name):
    try:
        value = headers.get(name)
    except Exception:
        return None
    if value is None:
        for key, candidate in dict(headers).items():
            if key.lower() == name.lower():
                return candidate
    return value


class FakeSession:
    def __init__(self, ws, query, args):
        self.ws = ws
        self.query = query
        self.args = args
        self.utterance = 0
        self.utterance_audio = 0.0
        self.last_partial = 0.0
        self.speech_open = False
        self.total_audio = 0.0
        self.received_bytes = 0
        self.dropped = False

    async def send(self, payload):
        await self.ws.send(json.dumps(payload))

    def validate(self):
        """Return an error message if the connection parameters are wrong."""
        key = header(request_headers(self.ws), "api-subscription-key")
        if not key:
            return "missing api-subscription-key header"
        language = (self.query.get("language_code") or [""])[0]
        if language not in VALID_LANGUAGES:
            return "invalid language_code %r" % language
        model = (self.query.get("model") or ["saaras:v3-realtime"])[0]
        if model not in VALID_MODELS:
            return "invalid model %r" % model
        mode = (self.query.get("mode") or ["transcribe"])[0]
        if mode not in VALID_MODES:
            return "invalid mode %r" % mode
        stream_type = (self.query.get("stream_type") or ["balanced"])[0]
        if stream_type not in {"fast", "balanced", "simulated"}:
            return "invalid stream_type %r" % stream_type
        if (self.query.get("encoding") or ["linear16"])[0] != "linear16":
            return "mock only speaks linear16"
        if (self.query.get("sample_rate") or ["16000"])[0] != "16000":
            return "mock only accepts sample_rate=16000"
        if (self.query.get("endpointing") or ["vad"])[0] not in {"vad", "manual"}:
            return "invalid endpointing"
        return None

    async def run(self):
        error = self.validate()
        if error:
            await self.send({"event": "error", "code": "invalid_request", "is_fatal": True,
                             "message": error, "status_code": 400})
            await self.ws.close(code=4000, reason=error[:100])
            log("  rejected connection: " + error)
            return

        await self.send({
            "event": "session.begin",
            "request_id": "mock-request-0001",
            "config": {
                "language_code": (self.query.get("language_code") or ["auto"])[0],
                "model": (self.query.get("model") or ["saaras:v3-realtime"])[0],
                "mode": (self.query.get("mode") or ["transcribe"])[0],
                "stream_type": (self.query.get("stream_type") or ["balanced"])[0],
                "turn_detection": "vad",
                "encoding": "linear16",
                "sample_rate": 16000,
            },
        })
        log("  session.begin sent")

        try:
            async for raw in self.ws:
                try:
                    message = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                event = message.get("event")
                if event == "audio_input":
                    try:
                        pcm = base64.b64decode(message.get("audio") or "")
                    except Exception:
                        pcm = b""
                    await self.on_audio(len(pcm))
                elif event == "end":
                    await self.finish()
                    return
                elif event == "ping":
                    await self.send({"event": "pong"})
        except websockets.ConnectionClosed:
            pass

    async def on_audio(self, size):
        duration = size / 32000.0  # 16 kHz * 2 bytes
        self.received_bytes += size
        self.total_audio += duration

        if self.args.drop_after and self.total_audio >= self.args.drop_after:
            if not self.dropped:
                self.dropped = True
                log("  dropping the socket on purpose (%.1f s of audio)" % self.total_audio)
                await self.ws.close(code=1011, reason="mock drop")
            return

        if not self.speech_open:
            self.speech_open = True
            await self.send({"event": "vad.speech_start", "utterance_idx": self.utterance})

        self.utterance_audio += duration
        if self.utterance_audio - self.last_partial >= PARTIAL_EVERY:
            self.last_partial = self.utterance_audio
            words = LINES[self.utterance % len(LINES)].split(" ")
            fraction = min(1.0, self.utterance_audio / UTTERANCE_SECONDS)
            count = max(1, int(round(len(words) * fraction)))
            await self.send({
                "event": "transcript.partial",
                "utterance_idx": self.utterance,
                "text": " ".join(words[:count]),
                "language": "hi-IN",
            })

        if self.utterance_audio >= UTTERANCE_SECONDS:
            await self.close_utterance()

    async def close_utterance(self):
        text = LINES[self.utterance % len(LINES)]
        await self.send({
            "event": "transcript.final",
            "utterance_idx": self.utterance,
            "text": text,
            "language": "hi-IN",
            "language_confidence": "0.97",
        })
        await self.send({"event": "vad.speech_end", "utterance_idx": self.utterance})
        self.utterance += 1
        self.utterance_audio = 0.0
        self.last_partial = 0.0
        self.speech_open = False

    async def finish(self):
        if self.speech_open and self.utterance_audio > 0.2:
            await self.close_utterance()
        await self.send({
            "event": "session.end",
            "request_id": "mock-request-0001",
            "total_duration_s": "%.2f" % self.total_audio,
            "total_utterances": self.utterance,
            "audio_duration_s": "%.2f" % self.total_audio,
        })
        await self.ws.close(code=1000, reason="done")
        log("  session.end sent (%d utterances, %.1f s of audio)"
            % (self.utterance, self.total_audio))


def make_handler(args):
    async def handler(ws):
        path = request_path(ws)
        query = parse_qs(urlparse(path).query)
        if "/speech-to-text-realtime/ws" not in path:
            await ws.close(code=4000, reason="wrong path")
            return
        if args.fail:
            message = "mock: refusing the session with close code %d" % args.fail
            try:
                await ws.send(json.dumps({"event": "error", "code": "mock_failure",
                                          "is_fatal": True, "message": message,
                                          "status_code": 403}))
            except Exception:
                pass
            await ws.close(code=args.fail, reason=message[:100])
            log("  refused with close code %d" % args.fail)
            return
        log("  client connected (%s)" % path.split("?")[0])
        session = FakeSession(ws, query, args)
        await session.run()
        log("  client disconnected")

    return handler


async def serve(args):
    async with websockets.serve(make_handler(args), args.host, args.port):
        log("mock Sarvam socket listening on ws://%s:%d/speech-to-text-realtime/ws"
            % (args.host, args.port))
        log("(this is a fake: it replays canned text and does not listen to the audio)")
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description="Fake Sarvam realtime socket")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--drop-after", type=float, default=0.0,
                        help="close the socket after this many seconds of audio")
    parser.add_argument("--fail", type=int, default=0,
                        help="refuse every session with this WebSocket close code")
    args = parser.parse_args()
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        log("\nstopped")


if __name__ == "__main__":
    main()
