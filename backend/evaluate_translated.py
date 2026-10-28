"""
Evaluate in English: Hinglish references are translated into English, and the model
translates the speech into English itself; then the two English texts are compared.

    reference:  "चलिए sample impress open करते हैं"  ->  "Let's open the sample impress"
    model:      speech -> English  (Whisper task="translate", Sarvam mode="translate")

References are translated with Sarvam's text translation API (needs SARVAM_API_KEY),
once, and saved next to your CSV as transcripts_english.csv. Later runs reuse that file.
Machine translations contain mistakes: open the file and correct any bad lines; your
corrections are used from then on. If you already have English translations, put them in
that file yourself (same format: path,english text) and no API calls are made.

Metrics: WER and CER as in evaluate.py, plus the standard translation metrics:
    BLEU  overlap of 1-4 word sequences with the reference (0-100, higher is better)
    chrF  overlap of character sequences (0-100, higher is better); more forgiving of
          small word-form differences, usually the more reliable of the two
A sentence can be translated correctly in many ways, so expect WER to be high even for
good translations; compare engines using BLEU/chrF rather than WER here.

Usage (same options as evaluate.py):
    python evaluate_translated.py --csv transcripts.csv --audio clips.tar.gz --engine whisper --language hi
    python evaluate_translated.py --csv transcripts.csv --audio clips.tar.gz --engine sarvam --language hi

--language is the language SPOKEN in the clips (hi), not the output language: English output
comes from translate mode. Passing --language en tells Whisper the audio is English.
"""

import argparse
import os
from pathlib import Path

import requests
import sacrebleu

from evaluate import add_dataset_args, convert_references, load_dataset, run_evaluation, save_and_summarize
from transcribers import get_transcriber

TRANSLATE_MODEL = "mayura:v1"


def translate_to_english(text):
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise SystemExit("Translating references needs SARVAM_API_KEY (in backend/.env)")
    response = requests.post(
        "https://api.sarvam.ai/translate",
        headers={"api-subscription-key": api_key},
        json={
            "input": text,
            "source_language_code": "auto",
            "target_language_code": "en-IN",
            "model": TRANSLATE_MODEL,
        },
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Sarvam translate error {response.status_code}: {response.text}")
    return response.json()["translated_text"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate speech translation into English")
    add_dataset_args(parser)
    args = parser.parse_args()

    items = load_dataset(args.csv, args.audio, args.language)[: args.limit]
    cache = Path(args.csv).with_name(f"{Path(args.csv).stem}_english.csv")
    convert_references(items, translate_to_english, cache)

    print(f"\nLoaded {len(items)} files. Engine: {args.engine} (translate to English)\n")
    rows = run_evaluation(items, get_transcriber(args.engine), task="translate")
    save_and_summarize(rows, args.out)

    scored = [r for r in rows if r["ref_norm"]]
    hypotheses = [r["hyp_norm"] for r in scored]
    references = [[r["ref_norm"] for r in scored]]
    bleu = sacrebleu.corpus_bleu(hypotheses, references).score
    chrf = sacrebleu.corpus_chrf(hypotheses, references).score
    print(f"BLEU {bleu:.1f}   chrF {chrf:.1f}   (0-100, higher is better)")


if __name__ == "__main__":
    main()
