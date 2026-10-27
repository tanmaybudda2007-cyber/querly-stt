"""
Test an engine against your own dataset of audio clips and reference transcripts.

Expected dataset:
  - a CSV where each line is   path/to/clip.wav, transcript
    (a header line such as "path,transcript" is fine; it is skipped automatically)
  - the audio clips, either as a .tar.gz archive or as an already-extracted folder.
    Paths in the CSV are looked up inside it.

Usage:
  python evaluate.py --csv transcripts.csv --audio clips.tar.gz --engine whisper
  python evaluate.py --csv transcripts.csv --audio clips.tar.gz --engine sarvam
  python evaluate.py --csv transcripts.csv --audio clips/ --engine whisper --language hi
  python evaluate.py --csv transcripts.csv --audio clips.tar.gz --limit 20   # quick trial

A .tar.gz is extracted once, into a folder next to it (clips.tar.gz -> clips/).
Later runs reuse that folder.

Metrics:
  WER  word error rate      = (substitutions + deletions + insertions) / reference words
  CER  character error rate = the same, counted in characters
  RTF  real-time factor     = processing time / audio duration  (< 1 is faster than real time)

WER and CER are computed over the whole dataset (total errors / total reference
words), so long clips count more than short ones. Both transcripts are normalized
first (see normalize), so punctuation and formatting differences aren't counted
as errors.

Per-file results are saved to a CSV (default: results.csv).
"""

import argparse
import csv
import tarfile
import time
import unicodedata
from pathlib import Path

import jiwer

from transcribers import get_transcriber


def normalize(text):
    """Make reference and hypothesis comparable.

    - Unicode NFC: the same Indic letter can be stored in more than one way
      (e.g. "क़" as one code point, or "क" + nukta); NFC picks one.
    - Punctuation and symbols become spaces, including the danda "।" and "॥".
      Vowel signs and virama are kept (they are marks, not punctuation).
    - Zero-width joiners are removed; they change rendering but are used
      inconsistently by people and models.
    - Native digits become ASCII ("२०२४" -> "2024"); Latin text is lower-cased.
    """
    text = unicodedata.normalize("NFC", text)
    out = []
    for ch in text:
        category = unicodedata.category(ch)
        if ch in "\u200b\u200c\u200d":  # zero-width space / non-joiner / joiner
            continue
        if category[0] in "PS":  # punctuation or symbol
            out.append(" ")
        elif category == "Nd":  # a digit in any script
            out.append(str(unicodedata.digit(ch)))
        else:
            out.append(ch)
    return " ".join("".join(out).lower().split())


def extract_archive(archive):
    """Extract clips.tar.gz into clips/ next to it, unless that was already done."""
    archive = Path(archive)
    name = archive.name.removesuffix(".gz").removesuffix(".tgz").removesuffix(".tar")
    target = archive.parent / name
    if target.is_dir():
        print(f"Using already-extracted audio in {target}/")
        return target

    print(f"Extracting {archive.name} -> {target}/ (one time) ...")
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            # Refuse paths that would land outside the target folder ("../x", "/etc/x").
            if not (target / member.name).resolve().is_relative_to(target.resolve()):
                raise SystemExit(f"Unsafe path in archive: {member.name}")
        if hasattr(tarfile, "data_filter"):  # Python 3.12+ (and recent 3.10/3.11 patches)
            tar.extractall(target, filter="data")
        else:
            tar.extractall(target)
    return target


def read_transcripts(csv_path):
    """Read (audio path, transcript) pairs from the CSV.

    Only the first comma separates path and transcript, so transcripts may contain
    commas. A first line that doesn't name a .wav file is treated as a header.
    """
    pairs = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for line_no, row in enumerate(csv.reader(f), 1):
            if not row or not row[0].strip():
                continue
            path, text = row[0].strip(), ",".join(row[1:]).strip()
            if line_no == 1 and not path.lower().endswith(".wav"):
                continue  # header
            pairs.append((path, text))
    return pairs


def find_audio(path, audio_root, by_name):
    """Locate a CSV path inside the extracted audio.

    Archives often add or drop a top-level folder compared to the CSV
    (CSV says "dir/a.wav", archive contains "clips/dir/a.wav"), so we try the
    exact path first and then fall back to the file name.
    """
    exact = audio_root / path
    if exact.exists():
        return exact
    matches = by_name.get(Path(path).name, [])
    if len(matches) == 1:
        return matches[0]
    # Same file name in several folders: pick the one whose path ends with the CSV path.
    for candidate in matches:
        if candidate.as_posix().endswith(Path(path).as_posix()):
            return candidate
    return None


def load_dataset(csv_path, audio, language):
    """Return a list of {"audio", "text", "language"} dicts."""
    audio = Path(audio)
    audio_root = audio if audio.is_dir() else extract_archive(audio)

    by_name = {}
    for wav in audio_root.rglob("*.wav"):
        by_name.setdefault(wav.name, []).append(wav)

    items, missing = [], []
    for path, text in read_transcripts(csv_path):
        wav = find_audio(path, audio_root, by_name)
        if wav is None:
            missing.append(path)
        else:
            items.append({"path": path, "audio": wav, "text": text, "language": language})

    if missing:
        print(f"Warning: {len(missing)} CSV entries have no matching audio, e.g. {missing[:3]}")
    if not items:
        raise SystemExit("No CSV entries matched an audio file. Check the paths in the CSV.")
    return items


def score(references, hypotheses):
    """Dataset-level WER and CER. Empty references are skipped (WER is undefined there)."""
    pairs = [(r, h) for r, h in zip(references, hypotheses) if r]
    if not pairs:
        return None, None
    refs, hyps = zip(*pairs)
    return jiwer.wer(list(refs), list(hyps)), jiwer.cer(list(refs), list(hyps))


def main():
    parser = argparse.ArgumentParser(description="Evaluate transcription on a dataset")
    parser.add_argument("--csv", required=True, help="CSV of: path/to/clip.wav, transcript")
    parser.add_argument("--audio", required=True, help="clips .tar.gz, or an extracted folder")
    parser.add_argument("--engine", default="whisper", choices=["whisper", "sarvam"])
    parser.add_argument("--language", default=None,
                        help="language code for every file (e.g. hi); default: auto-detect")
    parser.add_argument("--limit", type=int, help="only use the first N clips (for a quick test)")
    parser.add_argument("--out", default="results.csv", help="per-file results CSV")
    args = parser.parse_args()

    items = load_dataset(args.csv, args.audio, args.language)
    if args.limit:
        items = items[: args.limit]
    print(f"Loaded {len(items)} files. Engine: {args.engine}\n")
    transcriber = get_transcriber(args.engine)

    rows = []
    for i, item in enumerate(items, 1):
        start = time.time()
        try:
            result = transcriber.transcribe(str(item["audio"]), item["language"])
            hypothesis, error = result["text"], ""
            duration, detected = result["duration"], result["language"]
        except Exception as e:  # keep going; a failed file scores as an empty transcript
            hypothesis, error, duration, detected = "", str(e), 0, ""
        seconds = time.time() - start

        ref, hyp = normalize(item["text"]), normalize(hypothesis)
        file_wer, file_cer = score([ref], [hyp])
        rows.append({
            "file": item["path"],
            "language": item["language"] or detected,
            "reference": item["text"],
            "hypothesis": hypothesis,
            "ref_norm": ref,
            "hyp_norm": hyp,
            "wer": "" if file_wer is None else round(file_wer, 4),
            "cer": "" if file_cer is None else round(file_cer, 4),
            "duration": duration,
            "seconds": round(seconds, 2),
            "error": error,
        })
        status = f"ERROR {error}" if error else f"WER {file_wer:.2%}" if file_wer is not None else ""
        print(f"[{i}/{len(items)}] {item['path']}  {status}")

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Summary: overall, then per language if there is more than one.
    groups = {"ALL": rows}
    languages = sorted({r["language"] or "unknown" for r in rows})
    if len(languages) > 1:
        for lang in languages:
            groups[lang] = [r for r in rows if (r["language"] or "unknown") == lang]

    print(f"\n{'':10}{'files':>7}{'WER':>10}{'CER':>10}{'RTF':>8}")
    for name, group in groups.items():
        wer, cer = score([r["ref_norm"] for r in group], [r["hyp_norm"] for r in group])
        audio = sum(r["duration"] for r in group)
        rtf = sum(r["seconds"] for r in group) / audio if audio else 0
        wer_s = f"{wer:.2%}" if wer is not None else "-"
        cer_s = f"{cer:.2%}" if cer is not None else "-"
        print(f"{name:10}{len(group):>7}{wer_s:>10}{cer_s:>10}{rtf:>8.2f}")

    failures = sum(1 for r in rows if r["error"])
    if failures:
        print(f"\n{failures} file(s) failed - see the 'error' column.")
    print(f"\nPer-file results saved to {args.out}")


if __name__ == "__main__":
    main()
