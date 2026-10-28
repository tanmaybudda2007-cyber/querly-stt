"""
Evaluate in Latin script: Hinglish references are romanized (Hindi written in English
letters) and compared with the model output, romanized the same way.

Why: your references mix scripts. The same English word appears as "slide" in one
place and "स्लाइड" in another, and a model can't know which one the annotator chose.
Romanizing BOTH the reference and the model output removes most of that difference, so
the score reflects what was heard more than which script it was written in. It isn't
perfect: an English word written in Devanagari comes out close to, but not the same as,
its English spelling ("स्लाइड" -> "slaid" vs "slide"). CER counts that as a small error,
WER as a whole wrong word, so CER is the better guide here.

    reference:  "चलिए sample impress open करते हैं"  ->  "calie sample impres open karate haim"
    model:      Devanagari output is romanized the same way; Latin output is left as is.

Romanized Hindi has no standard spelling ("jaa"/"ja", "hoon"/"hun"), so letters
repeated in a row are collapsed on both sides before scoring ("jaa" -> "ja").

Romanizers (--romanizer):
    uroman  (default) free and local, rule-based; spelling is literal ("svaagat")
    sarvam  Sarvam's transliteration API; more natural Hinglish spelling; needs SARVAM_API_KEY

Usage (same options as evaluate.py, plus --romanizer):
    python evaluate_romanized.py --csv transcripts.csv --audio clips.tar.gz --engine whisper --language hi
    python evaluate_romanized.py --csv transcripts.csv --audio clips.tar.gz --engine whisper --language en

--language is the language spoken in the clips. With --language hi the model writes Hindi
and its output is romanized here, which is the fairest test of recognition. With
--language en, Whisper is told the audio is English; it then usually translates to English
instead of writing out the Hindi, which won't match a romanized reference.

Romanized references are saved next to your CSV (transcripts_roman_uroman.csv) for inspection.
"""

import argparse
import os
import re
from pathlib import Path

import requests

from evaluate import add_dataset_args, convert_references, load_dataset, run_evaluation, save_and_summarize
from transcribers import get_transcriber


def make_uroman():
    import uroman

    romanizer = uroman.Uroman()  # loads its rules once
    return romanizer.romanize_string


def make_sarvam_romanizer():
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise SystemExit("--romanizer sarvam needs SARVAM_API_KEY (in backend/.env)")

    def romanize(text):
        response = requests.post(
            "https://api.sarvam.ai/transliterate",
            headers={"api-subscription-key": api_key},
            json={"input": text, "source_language_code": "auto", "target_language_code": "en-IN"},
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Sarvam transliterate error {response.status_code}: {response.text}")
        return response.json()["transliterated_text"]

    return romanize


def collapse_repeats(text):
    """'jaa rahaa huum' -> 'ja raha hum': repeated letters are spelling noise here."""
    return re.sub(r"(.)\1+", r"\1", text)


def main():
    parser = argparse.ArgumentParser(description="Evaluate with romanized (Latin-script) text")
    add_dataset_args(parser)
    parser.add_argument("--romanizer", default="uroman", choices=["uroman", "sarvam"])
    args = parser.parse_args()

    romanizer = make_uroman() if args.romanizer == "uroman" else make_sarvam_romanizer()

    def romanize(text):
        return collapse_repeats(romanizer(text))

    items = load_dataset(args.csv, args.audio, args.language)[: args.limit]
    cache = Path(args.csv).with_name(f"{Path(args.csv).stem}_roman_{args.romanizer}.csv")
    convert_references(items, romanize, cache)

    print(f"\nLoaded {len(items)} files. Engine: {args.engine}, romanizer: {args.romanizer}\n")
    rows = run_evaluation(items, get_transcriber(args.engine), convert_hypothesis=romanize)
    save_and_summarize(rows, args.out)


if __name__ == "__main__":
    main()
