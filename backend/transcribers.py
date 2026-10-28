"""
Two interchangeable speech-to-text engines:

    WhisperTranscriber  - runs faster-whisper locally
    SarvamTranscriber   - calls the Sarvam AI speech-to-text API

Both have the same method:

    transcribe(wav_path, language=None, task="transcribe")
        -> {"text", "language", "duration", "segments"}

language is the language SPOKEN in the audio: None means "detect it automatically",
otherwise pass a code from LANGUAGES.
task="transcribe" writes down what was said; task="translate" outputs English instead,
using each engine's built-in speech translation.
"""

import io
import os
from math import gcd

import numpy as np
import requests
import soundfile as sf
from dotenv import load_dotenv
from scipy.signal import resample_poly

load_dotenv()

SAMPLE_RATE = 16000  # both engines expect 16 kHz mono audio

# Languages offered in the app: code -> name.
LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
}

# Whisper supports all of the above except Odia.
WHISPER_LANGUAGES = [code for code in LANGUAGES if code != "or"]

# Sarvam uses BCP-47 style codes.
SARVAM_CODES = {
    "en": "en-IN", "hi": "hi-IN", "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN",
    "mr": "mr-IN", "gu": "gu-IN", "kn": "kn-IN", "ml": "ml-IN", "pa": "pa-IN",
    "or": "od-IN",
}


def load_audio(path):
    """Read a WAV file as 16 kHz mono float32 samples."""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # stereo -> mono
    if sr != SAMPLE_RATE:
        # Polyphase resampling filters out frequencies above 8 kHz before
        # downsampling, so they don't fold back into the speech band.
        g = gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return audio


def load_cuda_library():
    """Make NVIDIA's cuBLAS 12 library available to faster-whisper on a GPU.

    faster-whisper's GPU engine (CTranslate2) looks up "libcublas.so.12" by name the
    first time it transcribes. Many machines, Google Colab included, don't have that
    version on the library search path, so transcription fails with
    "Library libcublas.so.12 is not found". If the pip package nvidia-cublas-cu12 is
    installed, we load its copy now; CTranslate2 then finds the already-loaded library.
    """
    import ctypes
    import sys

    if not sys.platform.startswith("linux"):
        return
    try:
        ctypes.CDLL("libcublas.so.12")  # already installed system-wide
        return
    except OSError:
        pass
    try:
        import nvidia.cublas.lib
    except ImportError:
        raise RuntimeError(
            "A GPU was found but the CUDA 12 library libcublas.so.12 is missing. "
            "Install it with:  pip install nvidia-cublas-cu12   "
            "(or set WHISPER_DEVICE=cpu to run on the CPU)"
        ) from None
    folder = list(nvidia.cublas.lib.__path__)[0]
    ctypes.CDLL(os.path.join(folder, "libcublas.so.12"), mode=ctypes.RTLD_GLOBAL)


class WhisperTranscriber:
    name = "whisper"

    def __init__(self, model_size=None, device=None):
        from faster_whisper import WhisperModel

        model_size = model_size or os.getenv("WHISPER_MODEL", "large-v3")
        device = device or os.getenv("WHISPER_DEVICE", "auto")
        if device == "auto":
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        if device == "cuda":
            load_cuda_library()
        # float16 is standard on GPU; int8 is much faster on CPU with little accuracy loss.
        compute_type = "float16" if device == "cuda" else "int8"

        self.model_name = model_size
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, path, language=None, task="transcribe"):
        audio = load_audio(path)
        if language is None:
            language = self.detect_language(audio)

        segments, _ = self.model.transcribe(
            audio,
            language=language,
            task=task,  # "translate" makes Whisper output English
            beam_size=5,
            vad_filter=True,  # skip silence, where Whisper tends to invent text
            condition_on_previous_text=False,  # avoids repetition loops, common in Indic output
        )
        segments = [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segments
        ]
        return {
            "text": " ".join(s["text"] for s in segments),
            "language": language,
            "duration": round(len(audio) / SAMPLE_RATE, 2),
            "segments": segments,
        }

    def detect_language(self, audio):
        """Pick the most likely language among the ones this app supports.

        Whisper chooses among ~100 languages and often misfires on short or accented
        clips. A common case: Hindi speech labelled as Urdu (nearly the same spoken
        language) and then written in Urdu script. Limiting the choice to our list
        avoids that.
        """
        _, _, probabilities = self.model.detect_language(audio, vad_filter=True)
        allowed = [(code, p) for code, p in probabilities if code in WHISPER_LANGUAGES]
        return max(allowed, key=lambda item: item[1])[0]


class SarvamTranscriber:
    name = "sarvam"
    url = "https://api.sarvam.ai/speech-to-text"
    chunk_seconds = 25  # the API accepts at most 30 s of audio per request

    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY")
        if not self.api_key:
            raise RuntimeError("SARVAM_API_KEY is not set (add it to backend/.env)")
        self.model_name = model or os.getenv("SARVAM_MODEL", "saaras:v3")

    def transcribe(self, path, language=None, task="transcribe"):
        if task == "translate" and not self.model_name.startswith("saaras"):
            raise RuntimeError("Speech translation needs SARVAM_MODEL=saaras:v3")
        audio = load_audio(path)
        chunk_len = self.chunk_seconds * SAMPLE_RATE
        segments, detected = [], []

        # Longer recordings are sent in consecutive 25 s pieces.
        for start in range(0, len(audio), chunk_len):
            piece = audio[start : start + chunk_len]
            result = self._request(piece, language, task)
            if result.get("transcript"):
                segments.append({
                    "start": round(start / SAMPLE_RATE, 2),
                    "end": round((start + len(piece)) / SAMPLE_RATE, 2),
                    "text": result["transcript"].strip(),
                })
            if result.get("language_code"):
                detected.append(result["language_code"])

        if language is None and detected:
            most_common = max(set(detected), key=detected.count)
            to_short = {v: k for k, v in SARVAM_CODES.items()}
            language = to_short.get(most_common, most_common)

        return {
            "text": " ".join(s["text"] for s in segments),
            "language": language,
            "duration": round(len(audio) / SAMPLE_RATE, 2),
            "segments": segments,
        }

    def _request(self, samples, language, task):
        wav = io.BytesIO()
        sf.write(wav, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")

        data = {"model": self.model_name}
        if self.model_name.startswith("saaras"):
            data["mode"] = task  # "transcribe" or "translate" (output in English)
        if language:  # omit to let Sarvam detect the language
            data["language_code"] = SARVAM_CODES[language]

        response = requests.post(
            self.url,
            headers={"api-subscription-key": self.api_key},
            data=data,
            files={"file": ("audio.wav", wav.getvalue(), "audio/wav")},
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Sarvam API error {response.status_code}: {response.text}")
        return response.json()


ENGINES = {"whisper": WhisperTranscriber, "sarvam": SarvamTranscriber}
_loaded = {}


def get_transcriber(name):
    """Create each engine once and reuse it (loading Whisper takes several seconds)."""
    if name not in ENGINES:
        raise ValueError(f"Unknown engine '{name}'. Choose from: {', '.join(ENGINES)}")
    if name not in _loaded:
        _loaded[name] = ENGINES[name]()
    return _loaded[name]
