#!/usr/bin/env python3
"""
End-to-end self-test. Proves the page's path works without a Sarvam key.

It starts the fake Sarvam socket (mock_sarvam.py) and the real app (live_server.py) in
one process, connects the way the browser does, streams PCM, and checks what
comes back - including that every byte of audio reached the saved .wav, that a
dropped socket reconnects, and that a refused session produces a clear message.

    python selftest.py

You should see PASS for every check and "ALL CHECKS PASSED" at the end.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import tempfile
import time
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import websockets  # noqa: E402

import mock_sarvam  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    mark = "PASS" if ok else "FAIL"
    line = "  [%s] %s" % (mark, name)
    if detail and not ok:
        line += "\n         -> " + str(detail)
    print(line, flush=True)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def pcm_seconds(seconds, amplitude=0.15):
    """Speech-shaped noise: loud/quiet bursts, so it looks like talking."""
    import math
    import random

    total = int(16000 * seconds)
    frames = bytearray()
    random.seed(7)
    for i in range(total):
        envelope = 0.5 + 0.5 * math.sin(2 * math.pi * 0.7 * i / 16000.0)
        value = random.uniform(-1, 1) * amplitude * envelope
        sample = int(max(-1.0, min(1.0, value)) * 32767)
        frames += sample.to_bytes(2, "little", signed=True)
    return bytes(frames)


class Browser:
    """Mimics static/index.html: send start, then binary PCM frames."""

    def __init__(self, port):
        self.url = "ws://127.0.0.1:%d/ws" % port
        self.messages = []
        self.ws = None

    async def __aenter__(self):
        self.ws = await websockets.connect(self.url, max_size=2 ** 22)
        return self

    async def __aexit__(self, *exc):
        try:
            await self.ws.close()
        except Exception:
            pass

    async def send(self, payload):
        await self.ws.send(json.dumps(payload))

    async def collect(self, seconds, until=None):
        """Read messages for a while, optionally stopping on a message type."""
        end = time.time() + seconds
        while time.time() < end:
            try:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=max(0.01, end - time.time()))
            except asyncio.TimeoutError:
                break
            except websockets.ConnectionClosed:
                self.messages.append({"type": "__closed__"})
                break
            try:
                message = json.loads(raw)
            except (ValueError, TypeError):
                continue
            self.messages.append(message)
            if until and message.get("type") == until:
                break
        return self.messages

    def of(self, kind):
        return [m for m in self.messages if m.get("type") == kind]


async def stream(browser, pcm, frame_bytes=1600, realtime=False):
    for offset in range(0, len(pcm), frame_bytes):
        frame = pcm[offset:offset + frame_bytes]
        if len(frame) < frame_bytes:
            frame = frame + b"\x00" * (frame_bytes - len(frame))
        await browser.ws.send(frame)
        if realtime:
            await asyncio.sleep(frame_bytes / 32000.0)


async def scenario_happy(port):
    print("\n1. happy path: 9 s of audio, three utterances", flush=True)
    pcm = pcm_seconds(9.0)
    async with Browser(port) as browser:
        await browser.send({"type": "start", "language_code": "auto",
                            "mode": "codemix", "stream_type": "fast"})
        await browser.collect(3, until="started")
        started = browser.of("started")
        check("server answered 'started'", len(started) == 1,
              "got %d" % len(started))
        session_name = started[0].get("session") if started else None

        await stream(browser, pcm)
        await browser.collect(1.5)
        await browser.send({"type": "stop"})
        # Read until the server says it closed: the final line of a session
        # legitimately arrives together with the end of the session.
        await browser.collect(8, until="closed")
        partials = browser.of("partial")
        finals = browser.of("final")
        check("partial transcripts arrived", len(partials) >= 5, "got %d" % len(partials))
        check("final transcripts arrived", len(finals) >= 2, "got %d" % len(finals))
        if finals:
            idxs = [f.get("idx") for f in finals]
            check("utterance indexes increase", idxs == sorted(set(idxs)), idxs)
            check("final text is not empty", all(f.get("text") for f in finals),
                  [f.get("text") for f in finals])
        if partials:
            check("partials carry text", all(p.get("text") for p in partials),
                  "some partial had no text")
            first_idx = partials[0].get("idx")
            before = [p for p in partials if p.get("idx") == first_idx]
            growing = [len(p["text"]) for p in before]
            check("partial text grows within an utterance",
                  growing == sorted(growing) and len(set(growing)) > 1, growing)

        closed = browser.of("closed")
        check("server confirmed the session closed", len(closed) == 1,
              "got %d" % len(closed))

        if session_name:
            folder = HERE / "sessions" / session_name
            wav_path = folder / "audio.wav"
            check("session folder exists", folder.is_dir(), str(folder))
            if wav_path.is_file():
                with wave.open(str(wav_path), "rb") as handle:
                    frames = handle.getnframes()
                    rate = handle.getframerate()
                    channels = handle.getnchannels()
                    width = handle.getsampwidth()
                check("saved wav is 16 kHz mono 16-bit",
                      (rate, channels, width) == (16000, 1, 2),
                      (rate, channels, width))
                check("every byte of audio reached the wav",
                      frames * 2 == len(pcm),
                      "wav has %d bytes, sent %d" % (frames * 2, len(pcm)))
            else:
                check("saved wav exists", False, str(wav_path))
            text_path = folder / "transcript.txt"
            if text_path.is_file():
                saved = text_path.read_text(encoding="utf-8").strip().splitlines()
                check("transcript.txt matches the finals",
                      saved == [f["text"] for f in finals],
                      "%r vs %r" % (saved, [f["text"] for f in finals]))
            else:
                check("transcript.txt exists", False, str(text_path))
            check("transcript.jsonl exists",
                  (folder / "transcript.jsonl").is_file())


async def scenario_realtime(port):
    print("\n2. real-time pacing: does a partial arrive while still speaking?", flush=True)
    pcm = pcm_seconds(2.5)
    async with Browser(port) as browser:
        await browser.send({"type": "start", "language_code": "hi-IN",
                            "mode": "codemix", "stream_type": "fast"})
        await browser.collect(3, until="started")
        begin = time.time()
        await stream(browser, pcm, realtime=True)
        elapsed = time.time() - begin
        await browser.collect(1.5)
        partials = browser.of("partial")
        check("partial arrived during a 2.5 s utterance", len(partials) >= 2,
              "got %d in %.1f s" % (len(partials), elapsed))
        await browser.send({"type": "stop"})
        await browser.collect(5, until="closed")


async def scenario_reconnect(port):
    print("\n3. socket dropped mid-session: does it reconnect and continue?", flush=True)
    pcm = pcm_seconds(8.0)
    async with Browser(port) as browser:
        await browser.send({"type": "start", "language_code": "auto",
                            "mode": "codemix", "stream_type": "fast"})
        await browser.collect(3, until="started")
        await stream(browser, pcm, realtime=True)
        await browser.collect(4)
        statuses = [m.get("message", "") for m in browser.of("status")]
        finals = browser.of("final")
        check("reconnect was announced to the page",
              any("econnect" in s for s in statuses), statuses)
        check("transcript continued after the drop", len(finals) >= 2,
              "%d finals" % len(finals))
        await browser.send({"type": "stop"})
        await browser.collect(5, until="closed")


async def scenario_refused(port):
    print("\n4. Sarvam refuses the session: is the message clear?", flush=True)
    async with Browser(port) as browser:
        await browser.send({"type": "start", "language_code": "auto",
                            "mode": "codemix", "stream_type": "fast"})
        await browser.collect(6, until="error")
        errors = browser.of("error")
        check("an error reached the page", len(errors) >= 1, "got %d" % len(errors))
        if errors:
            message = errors[0].get("message", "")
            check("error text names the problem",
                  "1003" in message or "API key" in message or "quota" in message.lower(),
                  message)
            check("error is marked fatal", errors[0].get("fatal") is True, errors[0])


def unit_checks():
    print("\n5. unit checks", flush=True)
    import live_server as app_module

    saved = os.environ.get("SARVAM_WS_URL")
    os.environ["SARVAM_WS_URL"] = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
    try:
        url = app_module.build_url({
            "language_code": "hi-IN", "model": "saaras:v3-realtime", "stream_type": "fast",
            "mode": "codemix", "threshold": 0.3, "silence_ms": 400,
        })
    finally:
        if saved is None:
            os.environ.pop("SARVAM_WS_URL", None)
        else:
            os.environ["SARVAM_WS_URL"] = saved
    check("url uses the realtime endpoint",
          url.startswith("wss://api.sarvam.ai/speech-to-text-realtime/ws?"), url)
    check("url asks for 16 kHz linear16",
          "sample_rate=16000" in url and "encoding=linear16" in url, url)
    check("url asks for codemix + fast",
          "mode=codemix" in url and "stream_type=fast" in url, url)
    check("url sets silence_duration_ms",
          "silence_duration_ms=400" in url, url)

    cfg = app_module.client_config({"language_code": "klingon", "mode": "nope",
                                    "stream_type": "turbo", "model": "evil"})
    check("bad settings fall back to safe defaults",
          (cfg["language_code"], cfg["mode"], cfg["stream_type"], cfg["model"]) ==
          ("auto", "codemix", "fast", "saaras:v3-realtime"), cfg)
    cfg = app_module.client_config({"silence_ms": 99999})
    check("silence_ms is clamped", cfg["silence_ms"] == 2000, cfg["silence_ms"])

    event = app_module.map_event({"event": "transcript.final", "utterance_idx": 3,
                                 "text": "haan", "language": "hi-IN"})
    check("final event maps to the page protocol",
          event["type"] == "final" and event["idx"] == 3 and event["text"] == "haan", event)
    check("session.end maps to 'ended'",
          app_module.map_event({"event": "session.end", "audio_duration_s": "3.2"})["type"]
          == "ended")
    check("unknown events are ignored",
          app_module.map_event({"event": "pong"}) is None)

    class FakeStatusError(Exception):
        status_code = 403

    message = app_module.describe_connect_error(FakeStatusError("boom"))
    check("HTTP 403 is explained as a bad key",
          "API key" in message and "403" in message, message)
    message = app_module.describe_connect_error(ConnectionRefusedError("timed out"))
    check("other connect failures are still readable",
          "Sarvam" in message and "timed out" in message, message)

    check("close code 1003 has a human explanation",
          1003 in app_module.CLOSE_CODES and "API key" in app_module.CLOSE_CODES[1003],
          app_module.CLOSE_CODES.get(1003))


async def run(port, mock_port):
    import uvicorn

    import live_server as app_module

    os.environ["SARVAM_API_KEY"] = "test-key-not-real"
    os.environ["SARVAM_WS_URL"] = "ws://127.0.0.1:%d/speech-to-text-realtime/ws" % mock_port

    config = uvicorn.Config(app_module.app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.05)
    if not server.started:
        print("could not start the web server", flush=True)
        return 1

    unit_checks()

    scenarios = [
        ("happy", scenario_happy, {"drop_after": 0.0, "fail": 0}),
        ("realtime", scenario_realtime, {"drop_after": 0.0, "fail": 0}),
        ("reconnect", scenario_reconnect, {"drop_after": 4.0, "fail": 0}),
        ("refused", scenario_refused, {"drop_after": 0.0, "fail": 1003}),
    ]
    for name, function, options in scenarios:
        args = argparse.Namespace(host="127.0.0.1", port=mock_port,
                                  drop_after=options["drop_after"],
                                  fail=options["fail"])
        mock_server = await websockets.serve(
            mock_sarvam.make_handler(args), "127.0.0.1", mock_port)
        try:
            await function(port)
        except Exception as exc:
            check("scenario '%s' ran without crashing" % name, False, repr(exc))
        finally:
            mock_server.close()
            await mock_server.wait_closed()
            await asyncio.sleep(0.2)

    server.should_exit = True
    try:
        await asyncio.wait_for(server_task, timeout=10)
    except (asyncio.TimeoutError, Exception):
        pass
    return 0


def main():
    port, mock_port = free_port(), free_port()
    print("=" * 68)
    print("Querly Live self-test (no API key needed - uses the fake Sarvam socket)")
    print("=" * 68)
    with tempfile.TemporaryDirectory() as temp:
        os.chdir(HERE)
        code = asyncio.run(run(port, mock_port))
    failed = [name for name, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 68)
    if failed:
        print("%d of %d checks FAILED:" % (len(failed), len(RESULTS)))
        for name in failed:
            print("  - " + name)
        print("=" * 68)
        return 1
    print("ALL %d CHECKS PASSED" % len(RESULTS))
    print("=" * 68)
    return code


if __name__ == "__main__":
    sys.exit(main())
