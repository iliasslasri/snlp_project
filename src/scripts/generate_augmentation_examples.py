"""Generate one example of each augmentation from a LibriSpeech audio file.

Usage:
    uv run python src/scripts/generate_augmentation_examples.py [--input <path>] [--out_dir <dir>]

Outputs are saved as .wav files in the output directory (default: data/augmentation_examples/).
"""
import argparse
import glob
import os
import sys

import torch
import torchaudio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.dataset import AugmentationPipeline


SAMPLE_RATE = 16000

# Default config matching configs/quantization.yaml
DEFAULT_CONFIG = {
    "time_stretch": True,
    "pitch_shift": True,
    "reverberation": True,
    "noise": True,
    "rir_dir": "data/rirs",
    "echo": {"enabled": True, "volume_range": [0.1, 0.5], "duration_range": [0.1, 0.5]},
    "random_noise": {"enabled": True, "noise_std": 0.001},
    "pink_noise": {"enabled": True, "noise_std": 0.01},
    "lowpass_filter": {"enabled": True, "cutoff_freq": 5000},
    "highpass_filter": {"enabled": True, "cutoff_freq": 500},
    "bandpass_filter": {"enabled": True, "cutoff_freq_low": 300, "cutoff_freq_high": 8000},
    "smooth": {"enabled": True, "window_size_range": [2, 10]},
    "boost_audio": {"enabled": True, "amount": 20},
    "duck_audio": {"enabled": True, "amount": 20},
    "updownresample": {"enabled": True, "intermediate_freq": 32000},
}


def find_sample_audio(data_dir="data/LibriSpeech/train-clean-100"):
    """Find the first .flac file in the dataset."""
    files = sorted(glob.glob(os.path.join(data_dir, "**/*.flac"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No audio files found in {data_dir}")
    return files[0]


def load_audio(path, target_sr=SAMPLE_RATE, max_seconds=5):
    """Load an audio file trimmed to max_seconds."""
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    max_samples = target_sr * max_seconds
    if waveform.shape[-1] > max_samples:
        waveform = waveform[:, :max_samples]
    return waveform


def main():
    parser = argparse.ArgumentParser(description="Generate augmentation examples")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input audio file (default: auto-detect from LibriSpeech)")
    parser.add_argument("--out_dir", type=str, default="data/augmentation_examples",
                        help="Output directory")
    parser.add_argument("--noise_dir", type=str, default="noise_fullband",
                        help="Directory of noise files for the 'noise' augmentation")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load source audio
    input_path = args.input or find_sample_audio()
    print(f"Source audio: {input_path}")
    waveform = load_audio(input_path)

    # Save the clean original
    clean_path = os.path.join(args.out_dir, "00_clean.wav")
    torchaudio.save(clean_path, waveform, SAMPLE_RATE)
    print(f"  Saved: {clean_path}")

    # Build pipeline with all augmentations enabled
    pipeline = AugmentationPipeline(
        sample_rate=SAMPLE_RATE,
        config=DEFAULT_CONFIG,
        noise_dir=args.noise_dir,
    )

    # Map of (display name, method) for each augmentation
    augmentations = [
        ("time_stretch", pipeline.time_stretch),
        ("pitch_shift", pipeline.pitch_shift),
        ("reverberation", pipeline.add_reverberation),
        ("noise", pipeline.add_noise),
        ("echo", pipeline.echo),
        ("random_noise", pipeline.random_noise),
        ("pink_noise", pipeline.pink_noise_aug),
        ("lowpass_filter", pipeline.lowpass_filter),
        ("highpass_filter", pipeline.highpass_filter),
        ("bandpass_filter", pipeline.bandpass_filter),
        ("smooth", pipeline.smooth),
        ("boost_audio", pipeline.boost_audio),
        ("duck_audio", pipeline.duck_audio),
        ("updownresample", pipeline.updownresample),
    ]

    for i, (name, aug_fn) in enumerate(augmentations, start=1):
        try:
            with torch.no_grad():
                augmented = aug_fn(waveform.clone())
            out_path = os.path.join(args.out_dir, f"{i:02d}_{name}.wav")
            torchaudio.save(out_path, augmented, SAMPLE_RATE)
            print(f"  Saved: {out_path}")
        except Exception as e:
            print(f"  FAILED [{name}]: {e}")

    print(f"\nDone! {len(augmentations)} augmentation examples saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
