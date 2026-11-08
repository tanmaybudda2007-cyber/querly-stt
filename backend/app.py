"""
Transcription + LLM API.

    GET  /api/options      engines, languages, LLM providers and defaults for the web page
    POST /api/transcribe   form fields: file (.wav), engine (whisper|sarvam), language (auto|code)
    POST /api/llm          JSON: transcript, and optionally prompt, provider, model,
                           temperature, max_tokens, language

Live mode (called by querly-live, which does the transcription; see live_llm.py):
    GET  /api/live/options           modes (with their prompt templates), providers
    POST /api/live/start             JSON: mode, prompt, provider, model, temperature,
                                     max_tokens, update_every, recent_minutes -> session_id
    POST /api/live/{id}/transcript   JSON: text, start, end, language -> update_due
    POST /api/live/{id}/update       refresh the mode's output (null if nothing new)
    POST /api/live/{id}/question     JSON: question
    POST /api/live/{id}/final        final output; ends the session

Run:  uvicorn app:app --port 8000
"""

import os
import tempfile
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

import live_llm
from llm import PROVIDERS, default_settings, is_available, run_llm
from transcribers import LANGUAGES, SARVAM_CODES, WHISPER_LANGUAGES, get_transcriber

app = FastAPI(title="Querly STT")


@app.get("/api/options")
def options():
    return {
        "languages": LANGUAGES,
        "engines": {
            "whisper": {"available": True, "languages": WHISPER_LANGUAGES},
            "sarvam": {
                "available": bool(os.getenv("SARVAM_API_KEY")),
                "languages": list(SARVAM_CODES),
            },
        },
        "llm": {
            "defaults": default_settings(),
            "providers": {
                name: {"available": is_available(name)[0], "note": is_available(name)[1],
                       "default_model": config["model"]}
                for name, config in PROVIDERS.items()
            },
        },
    }


# A plain `def` (not `async def`) makes FastAPI run this in a worker thread,
# so a long transcription doesn't block other requests.
@app.post("/api/transcribe")
def transcribe(
    file: UploadFile = File(...),
    engine: str = Form("whisper"),
    language: str = Form("auto"),
):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(400, "Please upload a .wav file")
    lang = None if language == "auto" else language
    if lang is not None and lang not in LANGUAGES:
        raise HTTPException(400, f"Unknown language '{language}'")
    if engine == "whisper" and lang == "or":
        raise HTTPException(400, "Whisper does not support Odia. Use Sarvam.")

    # Save the upload to a temporary file so the audio library can read it.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(file.file.read())
        path = tmp.name

    try:
        transcriber = get_transcriber(engine)
        start = time.time()
        result = transcriber.transcribe(path, lang)
        result["engine"] = engine
        result["model"] = transcriber.model_name
        result["processing_seconds"] = round(time.time() - start, 2)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {e}")
    finally:
        os.remove(path)


class LLMRequest(BaseModel):
    transcript: str
    prompt: str | None = None  # None = the template in prompt.txt
    provider: str | None = None  # None = LLM_PROVIDER from .env
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    language: str | None = None


@app.post("/api/llm")
def llm(request: LLMRequest):
    if not request.transcript.strip():
        raise HTTPException(400, "The transcript is empty")
    try:
        return run_llm(**request.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"LLM request failed: {e}")


# ------------------------------------------------------------------ live mode


class LiveStart(BaseModel):
    mode: str = "summarize"
    prompt: str | None = None  # custom template; None = prompts/live_<mode>.txt
    provider: str | None = None  # None = LLM_PROVIDER from .env
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    update_every: float = 120  # seconds of speech between automatic updates; 0 = off
    recent_minutes: float = 4  # transcript kept word for word; older parts are summarized


class LiveLine(BaseModel):
    text: str
    start: float  # seconds from the start of the recording
    end: float
    language: str | None = None


class LiveQuestion(BaseModel):
    question: str


def live_session(session_id):
    session = live_llm.get_session(session_id)
    if session is None:
        # querly-live reacts to this by starting a new session and resending the transcript.
        raise HTTPException(404, "Unknown live session (the backend may have restarted)")
    return session


def run_live(action):
    try:
        return action()
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"LLM request failed: {e}")


@app.get("/api/live/options")
def live_options():
    return live_llm.options()


@app.post("/api/live/start")
def live_start(request: LiveStart):
    try:
        return {"session_id": live_llm.start_session(**request.model_dump())}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/live/{session_id}/transcript")
def live_transcript(session_id: str, line: LiveLine):
    due = live_session(session_id).add(line.text, line.start, line.end, line.language)
    return {"update_due": due}


@app.post("/api/live/{session_id}/update")
def live_update(session_id: str):
    session = live_session(session_id)
    print(f"Live {session_id}: updating ({session.mode})...", flush=True)
    return run_live(lambda: session.update())


@app.post("/api/live/{session_id}/question")
def live_question(session_id: str, request: LiveQuestion):
    session = live_session(session_id)
    print(f"Live {session_id}: answering a question...", flush=True)
    return run_live(lambda: session.ask(request.question))


@app.post("/api/live/{session_id}/final")
def live_final(session_id: str):
    session = live_session(session_id)
    print(f"Live {session_id}: writing the final output...", flush=True)
    try:
        return run_live(lambda: session.update(final=True))
    finally:
        live_llm.end_session(session_id)
