"""
Send a transcript to an LLM with your own prompt.

All supported providers speak the same "OpenAI-compatible" chat API, so one small
function handles them all; only the address, API key and default model differ:

    sarvam  Sarvam AI (sarvam-105b)                       needs SARVAM_API_KEY
    groq    Groq, free tier, open models (Llama, Qwen...)  needs GROQ_API_KEY   (console.groq.com)
    gemini  Google Gemini, free tier                       needs GEMINI_API_KEY (aistudio.google.com)
    ollama  Any open model running locally with Ollama     no key; run `ollama serve`

The prompt is a template: {transcript} is replaced by the transcript and {language}
by the spoken language's name. The default template is in prompt.txt.
Defaults for provider, model, temperature and max tokens come from .env (LLM_*).
"""

import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from transcribers import LANGUAGES

load_dotenv()

PROMPT_FILE = Path(__file__).with_name("prompt.txt")

PROVIDERS = {
    "sarvam": {
        "url": "https://api.sarvam.ai/v2",
        "key": "SARVAM_API_KEY",
        "model": "sarvam-105b",
        # sarvam-105b reasons before answering, and that reasoning counts toward
        # max_tokens; "low" keeps it short so the answer isn't cut off.
        "extra": {"reasoning_effort": "low"},
    },
    "groq": {
        "url": "https://api.groq.com/openai/v1",
        "key": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key": "GEMINI_API_KEY",
        "model": "gemini-3.8-flash",
    },
    "ollama": {
        "url": os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
        "key": None,
        "model": "gemma3",
    },
}


def default_settings():
    return {
        "provider": os.getenv("LLM_PROVIDER", "groq"),
        "model": os.getenv("LLM_MODEL", ""),  # empty = the provider's default model
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.3")),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2048")),
        "prompt": PROMPT_FILE.read_text(encoding="utf-8"),
    }


def is_available(provider):
    """(True, "") if the provider can be used, else (False, reason)."""
    key = PROVIDERS[provider]["key"]
    if key and not os.getenv(key):
        return False, f"set {key} in backend/.env"
    return True, ""


def fill_prompt(template, transcript, language=None):
    """Insert the transcript into the template. If the template has no {transcript}
    placeholder, the transcript is appended at the end.

    Plain replace (not str.format) so other braces in your prompt, e.g. a JSON
    example, are left alone.
    """
    language_name = LANGUAGES.get(language, language or "unknown")
    text = template.replace("{language}", language_name)
    if "{transcript}" in text:
        return text.replace("{transcript}", transcript)
    return f"{text.rstrip()}\n\n{transcript}"


def run_llm(transcript, prompt=None, provider=None, model=None, temperature=None,
            max_tokens=None, language=None):
    """Return {"output", "provider", "model", "seconds"}. Unset arguments use the defaults."""
    defaults = default_settings()
    provider = provider or defaults["provider"]
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider '{provider}'. Choose from: {', '.join(PROVIDERS)}")
    ok, reason = is_available(provider)
    if not ok:
        raise RuntimeError(f"LLM provider '{provider}' is not set up: {reason}")

    config = PROVIDERS[provider]
    model = model or defaults["model"] or config["model"]
    body = {
        "model": model,
        "messages": [{"role": "user", "content": fill_prompt(prompt or defaults["prompt"], transcript, language)}],
        "temperature": defaults["temperature"] if temperature is None else temperature,
        "max_tokens": max_tokens or defaults["max_tokens"],
        **config.get("extra", {}),
    }
    headers = {}
    if config["key"]:
        headers["Authorization"] = f"Bearer {os.getenv(config['key'])}"

    start = time.time()
    data = _post_with_retries(f"{config['url']}/chat/completions", headers, body)
    output = data["choices"][0]["message"].get("content") or ""
    # Some open models write their reasoning inline as <think>...</think>; drop it.
    output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()
    return {"output": output, "provider": provider, "model": model,
            "seconds": round(time.time() - start, 2)}


def _post_with_retries(url, headers, body, attempts=3):
    """Free tiers rate-limit (HTTP 429); wait and retry a few times before giving up."""
    for attempt in range(attempts):
        try:
            response = requests.post(url, headers=headers, json=body, timeout=180)
        except requests.ConnectionError:
            if "localhost" in url or "127.0.0.1" in url:
                raise RuntimeError(f"Can't reach Ollama at {url}. Is `ollama serve` running?") from None
            raise
        if response.status_code == 200:
            return response.json()
        if response.status_code in (429, 500, 502, 503) and attempt < attempts - 1:
            time.sleep(5 * 2**attempt)
            continue
        raise RuntimeError(f"LLM error {response.status_code}: {response.text[:500]}")
