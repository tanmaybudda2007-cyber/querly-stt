"""
Evaluate in English: Hinglish references are translated into English, and the model
translates the speech into English itself; then the two English texts are compared.

    reference:  "चलिए sample impress open करते हैं"  ->  "Let's open the sample impress"
    model:      speech -> English  (Whisper task="translate", Sarvam mode="translate")

Translators for the references (--translator):
    google  (default) Google Translate; free, no key, needs internet. Uses an unofficial
            free endpoint that rate-limits: requests are spaced 1 s apart and retried, and
            if it still gives up, rerun the same command; it continues where it stopped.
    nllb    Meta's NLLB-200 model, run locally; offline after a one-time ~2.5 GB download.
            Fast on a GPU (e.g. Colab), slow on a CPU. Needs torch (preinstalled on Colab).
    sarvam  Sarvam's translation API; needs SARVAM_API_KEY.

Each reference is translated once and saved next to your CSV, e.g.
transcripts_english_google.csv; later runs reuse that file. Machine translations contain
mistakes: open the file and correct any bad lines, and your corrections are used from then
on. If you already have English translations, save them under that name yourself (same
format: path,english text) and nothing is translated.

Metrics: WER and CER as in evaluate.py, plus the standard translation metrics:
    BLEU  overlap of 1-4 word sequences with the reference (0-100, higher is better)
    chrF  overlap of character sequences (0-100, higher is better); more forgiving of
          small word-form differences, usually the more reliable of the two
A sentence can be translated correctly in many ways, so expect WER to be high even for
good translations; compare engines using BLEU/chrF rather than WER here.

Usage (same options as evaluate.py, plus --translator):
    python evaluate_translated.py --csv transcripts.csv --audio clips.tar.gz --engine whisper --language hi
    python evaluate_translated.py --csv transcripts.csv --audio clips.tar.gz --engine sarvam --language hi --translator nllb

--language is the language SPOKEN in the clips (hi), not the output language: English output
comes from translate mode. Passing --language en tells Whisper the audio is English.
"""

import argparse
import importlib.util
import os
import time
from pathlib import Path

import requests
import sacrebleu

from evaluate import add_dataset_args, convert_references, load_dataset, run_evaluation, save_and_summarize
from transcribers import get_transcriber

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
# NLLB names languages by language + script.
NLLB_CODES = {
    "hi": "hin_Deva", "en": "eng_Latn", "bn": "ben_Beng", "ta": "tam_Taml", "te": "tel_Telu",
    "mr": "mar_Deva", "gu": "guj_Gujr", "kn": "kan_Knda", "ml": "mal_Mlym", "pa": "pan_Guru",
    "or": "ory_Orya",
}


def make_google_translator(pause=1.0, retries=5):
    """Google's free endpoint rate-limits bursts of requests, so we wait `pause` seconds
    between requests and, when a request fails, retry after 10 s, 20 s, 40 s, ..."""
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source="auto", target="en")

    def translate(text):
        for attempt in range(retries + 1):
            try:
                time.sleep(pause)
                return translator.translate(text)
            except Exception as e:
                if attempt == retries:
                    raise
                wait = 10 * 2**attempt
                print(f"  Google error ({type(e).__name__}); retrying in {wait} s")
                time.sleep(wait)

    return translate


def make_nllb_translator(language):
    missing = [m for m in ("torch", "transformers", "sentencepiece") if importlib.util.find_spec(m) is None]
    if missing:
        raise SystemExit(f"--translator nllb needs: pip install {' '.join(missing)}")
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {NLLB_MODEL} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL, src_lang=NLLB_CODES[language])
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL).to(device)
    english = tokenizer.convert_tokens_to_ids("eng_Latn")

    def translate(text):
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, forced_bos_token_id=english, max_new_tokens=256)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    return translate


def make_sarvam_translator():
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise SystemExit("--translator sarvam needs SARVAM_API_KEY (in backend/.env)")

    def translate(text):
        response = requests.post(
            "https://api.sarvam.ai/translate",
            headers={"api-subscription-key": api_key},
            json={
                "input": text,
                "source_language_code": "auto",
                "target_language_code": "en-IN",
                "model": "mayura:v1",
            },
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Sarvam translate error {response.status_code}: {response.text}")
        return response.json()["translated_text"]

    return translate


def main():
    parser = argparse.ArgumentParser(description="Evaluate speech translation into English")
    add_dataset_args(parser)
    parser.add_argument("--translator", default="google", choices=["google", "nllb", "sarvam"],
                        help="how to translate the reference transcripts (default: google)")
    args = parser.parse_args()

    items = load_dataset(args.csv, args.audio, args.language)[: args.limit]
    cache = Path(args.csv).with_name(f"{Path(args.csv).stem}_english_{args.translator}.csv")

    # Only set up a translator if some references still need translating
    # (avoids loading NLLB when everything is already in the cache file).
    translator = None

    def translate(text):
        nonlocal translator
        if translator is None:
            if args.translator == "google":
                translator = make_google_translator()
            elif args.translator == "nllb":
                translator = make_nllb_translator(args.language or "hi")
            else:
                translator = make_sarvam_translator()
        return translator(text)

    convert_references(items, translate, cache)

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
