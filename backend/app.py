"""
Transcription + LLM API.

    GET  /api/options      engines, languages, LLM providers and defaults for the web page
    POST /api/transcribe   form fields: file (.wav), engine (whisper|sarvam), language (auto|code)
    POST /api/llm          JSON: transcript, and optionally prompt, provider, model,
                           temperature, max_tokens, language

Run:  uvicorn app:app --port 8000
"""

import os
import tempfile
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from llm import PROVIDERS, default_settings, is_available, run_llm
from transcribers import LANGUAGES, SARVAM_CODES, WHISPER_LANGUAGES, get_transcriber

app = FastAPI(title="Indic Transcribe")


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
