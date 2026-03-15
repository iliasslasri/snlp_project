"""
Download a LibriSpeech split from HuggingFace (openslr/librispeech_asr) and
reconstruct the standard directory layout:

    LibriSpeech/<split>/<speaker_id>/<chapter_id>/<utt_id>.flac

Usage:
    uv run --with datasets python scripts/download_librispeech.py --split test-clean
    uv run --with datasets python scripts/download_librispeech.py --split test-other
    uv run --with datasets python scripts/download_librispeech.py --split dev-clean
"""

import argparse
import os

import io

import soundfile as sf
from datasets import Audio, load_dataset

HF_SPLIT_MAP = {
    "train-clean-100": ("clean", "train.100"),
    "train-clean-360": ("clean", "train.360"),
    "train-other-500": ("other", "train.500"),
    "dev-clean":       ("clean", "validation"),
    "dev-other":       ("other", "validation"),
    "test-clean":      ("clean", "test"),
    "test-other":      ("other", "test"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test-clean", choices=list(HF_SPLIT_MAP))
    parser.add_argument("--out-dir", default="LibriSpeech", help="Root output directory")
    parser.add_argument("--cache-dir", default=None, help="HuggingFace cache directory")
    args = parser.parse_args()

    subset, hf_split = HF_SPLIT_MAP[args.split]
    out_root = os.path.join(args.out_dir, args.split)

    print(f"Downloading '{args.split}' (subset='{subset}', hf_split='{hf_split}') …")
    ds = load_dataset(
        "openslr/librispeech_asr",
        subset,
        split=hf_split,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    # Disable automatic audio decoding — read raw bytes with soundfile instead
    # (avoids the torchcodec / FFmpeg dependency on Windows)
    ds = ds.cast_column("audio", Audio(decode=False))

    print(f"  {len(ds)} utterances → {out_root}/")
    for i, item in enumerate(ds):
        speaker  = str(item["speaker_id"])
        chapter  = str(item["chapter_id"])
        utt_id   = item["id"]          # e.g. "1284-134647-0000"
        raw      = item["audio"]["bytes"]

        audio, sr = sf.read(io.BytesIO(raw))

        out_dir = os.path.join(out_root, speaker, chapter)
        os.makedirs(out_dir, exist_ok=True)
        sf.write(os.path.join(out_dir, f"{utt_id}.flac"), audio, sr, subtype="PCM_16")

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(ds)}")

    print(f"Done — {len(ds)} files written to {out_root}/")


if __name__ == "__main__":
    main()
