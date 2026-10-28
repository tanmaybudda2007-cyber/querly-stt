"""
Transcribe audio files and run each transcript through an LLM with your prompt.
This is the command-line version of the web app's pipeline, for batches and Colab.

Usage:
    python process.py recording.wav
    python process.py clips/ --engine whisper --language hi --provider groq
    python process.py a.wav b.wav --provider ollama --model gemma3 --prompt my_prompt.txt
    python process.py clips/ --provider gemini --temperature 0 --max-tokens 1024

Anything you don't pass comes from backend/.env (LLM_PROVIDER, LLM_MODEL, ...) and the
prompt from prompt.txt. Each result is printed and appended to outputs.jsonl (one JSON
object per line: file, language, transcript, llm_output, provider, model). If the LLM
step fails, the transcript is still saved, with an "error" field.
"""

import argparse
import json
from pathlib import Path

from llm import PROVIDERS, run_llm
from transcribers import get_transcriber


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio, then process it with an LLM")
    parser.add_argument("inputs", nargs="+", help=".wav files and/or folders containing .wav files")
    parser.add_argument("--engine", default="whisper", choices=["whisper", "sarvam"])
    parser.add_argument("--language", default=None, help="language spoken (e.g. hi); default: auto-detect")
    parser.add_argument("--provider", choices=list(PROVIDERS), help="LLM provider (default: LLM_PROVIDER)")
    parser.add_argument("--model", help="LLM model (default: LLM_MODEL or the provider's default)")
    parser.add_argument("--prompt", help="prompt template file (default: prompt.txt)")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--out", default="outputs.jsonl")
    args = parser.parse_args()

    files = []
    for item in map(Path, args.inputs):
        files += sorted(item.rglob("*.wav")) if item.is_dir() else [item]
    if not files:
        raise SystemExit("No .wav files found")
    prompt = Path(args.prompt).read_text(encoding="utf-8") if args.prompt else None

    transcriber = get_transcriber(args.engine)
    with open(args.out, "a", encoding="utf-8") as out:
        for i, path in enumerate(files, 1):
            print(f"\n=== [{i}/{len(files)}] {path}")
            try:
                result = transcriber.transcribe(str(path), args.language)
            except Exception as e:  # keep going with the other files
                print(f"Transcription ERROR: {e}")
                continue
            print(f"Transcript ({result['language']}): {result['text']}\n")
            record = {"file": str(path), "language": result["language"], "transcript": result["text"]}

            try:
                answer = run_llm(result["text"], prompt=prompt, provider=args.provider,
                                 model=args.model, temperature=args.temperature,
                                 max_tokens=args.max_tokens, language=result["language"])
                print(f"LLM ({answer['provider']}, {answer['model']}):\n{answer['output']}")
                record.update(llm_output=answer["output"], provider=answer["provider"], model=answer["model"])
            except Exception as e:  # still save the transcript, so it isn't lost
                print(f"LLM ERROR: {e}")
                record["error"] = str(e)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
