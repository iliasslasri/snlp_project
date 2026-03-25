"""
Streaming K-Means training on LibriSpeech-960h using HuBERT layer-N features.

Streams audio from HuggingFace `datasets`, extracts hidden-state
features with the speech_encoder HuBERT wrapper, and fits sklearn MiniBatchKMeans
incrementally via partial_fit().

The saved model is a raw sklearn KMeans-compatible object serialized with joblib,
directly loadable by `speech_encoder.KMeansQuantizer.from_pretrained(path)`.

Usage:
    uv run python scripts/train_kmeans.py                           # defaults
    uv run python scripts/train_kmeans.py kmeans.n_clusters=200     # override clusters
    uv run python scripts/train_kmeans.py kmeans.max_samples=100    # quick test
"""

import logging
import os
import sys
import time

import hydra
import joblib
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.cluster import MiniBatchKMeans

# ---------------------------------------------------------------------------
# Add speech_encoder to path so we can import it
# ---------------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "speech_encoder", "src"))

from speech_encoder import HuBERT  # noqa: E402


def stream_librispeech():
    """Yield audio samples from LibriSpeech-960h via HuggingFace datasets in streaming mode.

    Iterates over all three training splits that constitute the 960-hour set:
    train.clean.100, train.clean.360, and train.other.500.

    Each yielded item has item["audio"]["array"] (numpy float32) and
    item["audio"]["sampling_rate"] (int, always 16 000 for LibriSpeech).
    """
    import io

    import soundfile as sf
    from datasets import Audio, load_dataset

    splits = ["train.clean.100", "train.clean.360", "train.other.500"]
    for split in splits:
        # Disable automatic audio decoding (avoids torchcodec)
        ds = load_dataset(
            "openslr/librispeech_asr",
            split=split,
            streaming=True,
        ).cast_column("audio", Audio(decode=False))

        for item in ds:
            audio_bytes = item["audio"]["bytes"]
            audio_data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            yield {
                **item,
                "audio": {"array": audio_data, "sampling_rate": sr},
            }


def batched(iterable, n):
    """Batch an iterable into chunks of size n (last chunk may be shorter)."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


@torch.inference_mode()
def extract_features(
    hubert: HuBERT,
    waveforms: list[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    """Extract HuBERT hidden-state features for a list of waveforms.

    Returns a 2-D numpy array of shape (total_frames, dim).
    """
    # Pad waveforms to equal length for batched inference
    lengths = [len(w) for w in waveforms]
    max_len = max(lengths)
    padded = np.zeros((len(waveforms), max_len), dtype=np.float32)
    for i, w in enumerate(waveforms):
        padded[i, : len(w)] = w

    wav_tensor = torch.from_numpy(padded).to(device)
    len_tensor = torch.tensor(lengths, dtype=torch.long, device=device)

    hidden_states, out_lengths = hubert(wav_tensor, len_tensor)

    # Gather valid frames (discard padding)
    all_frames = []
    for i in range(hidden_states.shape[0]):
        valid_len = out_lengths[i].item() if out_lengths is not None else hidden_states.shape[1]
        all_frames.append(hidden_states[i, :valid_len].cpu().numpy())

    return np.concatenate(all_frames, axis=0)


def train_kmeans(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # Load HuBERT
    logging.info(f"Loading HuBERT '{cfg.model.name}' (layer {cfg.model.layer})...")
    hubert = HuBERT.from_pretrained(cfg.model.name, layer=cfg.model.layer).to(device)
    hubert.eval()
    logging.info("HuBERT loaded.")

    # MiniBatchKMeans (supports partial_fit for streaming)
    kmeans = MiniBatchKMeans(
        n_clusters=cfg.kmeans.n_clusters,
        random_state=cfg.kmeans.seed,
        batch_size=1024,  # internal sklearn mini-batch size (frames)
        verbose=0,
    )
    logging.info(f"MiniBatchKMeans initialized with n_clusters={cfg.kmeans.n_clusters}")

    # Streaming training loop
    max_samples = cfg.kmeans.get("max_samples", None)
    batch_size = cfg.kmeans.batch_size
    log_interval = cfg.kmeans.get("log_interval", 50)

    total_samples = 0
    total_frames = 0
    batch_idx = 0
    t_start = time.time()

    logging.info("Starting streaming K-Means training on LibriSpeech-960h...")

    for audio_batch in batched(stream_librispeech(), batch_size):
        # Check early stopping
        if max_samples is not None and total_samples >= max_samples:
            logging.info(f"Reached max_samples={max_samples}, stopping.")
            break

        # Trim batch if we'd exceed max_samples
        if max_samples is not None:
            remaining = max_samples - total_samples
            audio_batch = audio_batch[:remaining]

        # Extract raw waveforms (all at 16kHz for LibriSpeech)
        waveforms = [item["audio"]["array"].astype(np.float32) for item in audio_batch]

        # Extract HuBERT features
        features = extract_features(hubert, waveforms, device)  # (N_frames, 768)

        # Incremental fit
        kmeans.partial_fit(features)

        total_samples += len(waveforms)
        total_frames += features.shape[0]
        batch_idx += 1

        if batch_idx % log_interval == 0:
            elapsed = time.time() - t_start
            logging.info(
                f"Batch {batch_idx} | "
                f"Samples: {total_samples} | "
                f"Frames: {total_frames:,} | "
                f"Inertia: {kmeans.inertia_:.2f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    elapsed = time.time() - t_start
    logging.info(
        f"Training complete. "
        f"Total samples: {total_samples}, Total frames: {total_frames:,}, "
        f"Final inertia: {kmeans.inertia_:.2f}, Time: {elapsed:.1f}s"
    )

    # Save the model
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    save_path = os.path.join(output_dir, cfg.output.save_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    joblib.dump(kmeans, save_path)
    logging.info(f"K-Means model saved to: {save_path}")
    logging.info(
        f"Load with: KMeansQuantizer.from_pretrained('{save_path}')"
    )


@hydra.main(version_base=None, config_path="../configs", config_name="kmeans")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    train_kmeans(cfg)


if __name__ == "__main__":
    main()
