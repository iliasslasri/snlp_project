"""High-level evaluation classes for GSLM model evaluation.

Classes:
    UEDEvaluator      – Unit Edit Distance per augmentation type
    ABXEvaluator      – ABX phoneme discrimination (uses fastabx)
    LexicalEvaluator  – sWUGGY (real-word vs pseudo-word preference)
    SyntacticEvaluator – sBLIMP (grammatical vs ungrammatical preference)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "speech_encoder", "src"))

from .dataset import AudioDataset
from .metrics import NgramLM, batch_unit_edit_distance, sblim_score, swuggy_score
from .models import RobustQuantizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unit Edit Distance Evaluator
# ---------------------------------------------------------------------------


class UEDEvaluator:
    """Evaluates Unit Edit Distance (UED) for E1 across augmentation types.

    Encodes clean and augmented audio with E1 (the trained MLP on top of HuBERT)
    and computes UED between the resulting unit sequences.
    """

    def __init__(
        self,
        upstream_encoder: torch.nn.Module,
        e1_model: RobustQuantizer,
        device: torch.device,
        vocab_size: int = 500,
    ):
        """
        Args:
            upstream_encoder: HuBERT dense encoder (E0.dense), frozen.
            e1_model:         Trained RobustQuantizer (MLP).
            device:           Torch device.
            vocab_size:       Number of discrete units (blank token = vocab_size).
        """
        self.upstream = upstream_encoder
        self.e1 = e1_model
        self.device = device
        self.vocab_size = vocab_size

    @torch.no_grad()
    def encode(
        self,
        waveforms: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Encode a batch of waveforms to deduplicated unit sequences using E1.

        Args:
            waveforms: [batch, time] float tensor at 16 kHz.
            lengths:   [batch] integer tensor of sample lengths (or None).

        Returns:
            List of unit sequences (one per waveform), with blanks filtered out
            and consecutive duplicates removed.
        """
        feats, out_lens = self.upstream(waveforms, lengths)
        logits = self.e1(feats)          # [batch, seq, vocab+1]
        preds = logits.argmax(dim=-1)    # [batch, seq]

        if out_lens is None:
            out_lens = torch.full(
                (preds.shape[0],), preds.shape[1], dtype=torch.long, device=self.device
            )

        units: List[List[int]] = []
        for i in range(preds.shape[0]):
            valid = preds[i, : out_lens[i]]
            # Filter blank first, then deduplicate — avoids [1, blank, 1] → [1, 1] edge case
            no_blank = valid[valid != self.vocab_size]
            deduped = torch.unique_consecutive(no_blank).tolist()
            units.append(deduped)
        return units

    # ------------------------------------------------------------------
    def evaluate_dataset(
        self,
        dataset: AudioDataset,
        batch_size: int = 8,
        num_workers: int = 0,
    ) -> Dict[str, float]:
        """Evaluate UED on a dataset that yields (clean, augmented) pairs.

        Returns:
            Dict with keys 'mean_ued' and 'std_ued'.
        """
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=_paired_collate_fn,
            num_workers=num_workers,
            shuffle=False,
        )

        self.e1.eval()
        self.upstream.eval()
        all_ueds: List[float] = []

        for clean_audio, clean_lens, aug_audio, aug_lens in loader:
            clean_audio = clean_audio.to(self.device)
            clean_lens = clean_lens.to(self.device)
            aug_audio = aug_audio.to(self.device)
            aug_lens = aug_lens.to(self.device)

            clean_units = self.encode(clean_audio, clean_lens)
            aug_units = self.encode(aug_audio, aug_lens)
            _, scores = batch_unit_edit_distance(clean_units, aug_units)
            all_ueds.extend(scores)

        n = len(all_ueds)
        std = float(np.std(all_ueds))
        return {
            "mean_ued": float(np.mean(all_ueds)),
            "std_ued": std,
            "sem_ued": std / np.sqrt(n) if n > 1 else 0.0,
            "n": n,
        }

    def evaluate_augmentation(
        self,
        root: str,
        splits: Union[str, List[str]],
        augmentation_name: str,
        base_aug_config: DictConfig,
        noise_dir: Optional[str] = None,
        max_length: Optional[int] = None,
        n_samples: Optional[int] = None,
        batch_size: int = 8,
        num_workers: int = 0,
    ) -> Dict[str, float]:
        """Evaluate UED for a single augmentation type.

        Builds a dataset with only the specified augmentation enabled, then
        calls :meth:`evaluate_dataset`.

        Args:
            root:               LibriSpeech root directory.
            splits:             Dataset split(s) (e.g. 'test-clean' or ['test-clean', 'test-other']).
            augmentation_name:  Key of the augmentation to isolate
                                (e.g. 'time_stretch', 'pitch_shift',
                                'reverberation', 'noise').
            base_aug_config:    Full augmentation config from Hydra.
            noise_dir:          Path to noise files (needed for 'noise' aug).
            max_length:         Max audio length in samples.
            n_samples:          If given, subsample the dataset randomly.
            batch_size:         Batch size for the data loader.
            num_workers:        DataLoader workers.

        Returns:
            Dict with 'mean_ued' and 'std_ued'.
        """
        # Handle both single string and list of splits
        if isinstance(splits, str):
            splits = [splits]

        aug_config = _build_single_aug_config(augmentation_name, base_aug_config)

        # Create datasets for each split and combine them
        datasets = []
        for split in splits:
            dataset = AudioDataset(
                root=root,
                split=split,
                augment=True,
                config=aug_config,
                noise_dir=noise_dir,
                max_length=max_length,
            )
            datasets.append(dataset)

        # Combine all datasets
        from torch.utils.data import ConcatDataset
        combined_dataset: Union[ConcatDataset, Subset] = ConcatDataset(datasets)

        if n_samples is not None and n_samples < len(combined_dataset):
            indices = torch.randperm(len(combined_dataset))[:n_samples].tolist()
            combined_dataset = Subset(combined_dataset, indices)

        logger.info(
            f"Evaluating UED for '{augmentation_name}' on {len(combined_dataset)} samples "
            f"from split(s): {'+'.join(splits)} …"
        )
        return self.evaluate_dataset(combined_dataset, batch_size=batch_size, num_workers=num_workers)


# ---------------------------------------------------------------------------
# Baseline UED Evaluator (E0: HuBERT + KMeans, no E1)
# ---------------------------------------------------------------------------


class BaselineUEDEvaluator:
    """Evaluates UED using E0 (HuBERT + KMeans) without the learned E1 quantizer.

    Both clean and augmented audio are encoded by E0 directly.
    This serves as the reference row in evaluation tables.
    """

    def __init__(self, e0: torch.nn.Module, device: torch.device):
        self.e0 = e0
        self.device = device

    @torch.no_grad()
    def encode(
        self,
        waveforms: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        outputs = self.e0(waveforms, lengths=lengths, formatted=True)
        return [o["units"] for o in outputs]

    def evaluate_dataset(self, dataset, batch_size: int = 8, num_workers: int = 0) -> Dict[str, float]:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=_paired_collate_fn,
            num_workers=num_workers,
            shuffle=False,
        )
        self.e0.eval()
        all_ueds: List[float] = []
        for clean_audio, clean_lens, aug_audio, aug_lens in loader:
            clean_audio = clean_audio.to(self.device)
            clean_lens = clean_lens.to(self.device)
            aug_audio = aug_audio.to(self.device)
            aug_lens = aug_lens.to(self.device)
            clean_units = self.encode(clean_audio, clean_lens)
            aug_units = self.encode(aug_audio, aug_lens)
            _, scores = batch_unit_edit_distance(clean_units, aug_units)
            all_ueds.extend(scores)
        n = len(all_ueds)
        std = float(np.std(all_ueds))
        return {"mean_ued": float(np.mean(all_ueds)), "std_ued": std, "sem_ued": std / np.sqrt(n) if n > 1 else 0.0, "n": n}

    def evaluate_augmentation(
        self,
        root: str,
        splits: Union[str, List[str]],
        augmentation_name: str,
        base_aug_config: DictConfig,
        noise_dir: Optional[str] = None,
        max_length: Optional[int] = None,
        n_samples: Optional[int] = None,
        batch_size: int = 8,
        num_workers: int = 0,
    ) -> Dict[str, float]:
        # Handle both single string and list of splits
        if isinstance(splits, str):
            splits = [splits]

        aug_config = _build_single_aug_config(augmentation_name, base_aug_config)

        # Create datasets for each split and combine them
        datasets = []
        for split in splits:
            dataset = AudioDataset(
                root=root,
                split=split,
                augment=True,
                config=aug_config,
                noise_dir=noise_dir,
                max_length=max_length,
            )
            datasets.append(dataset)

        # Combine all datasets
        from torch.utils.data import ConcatDataset
        combined_dataset: Union[ConcatDataset, Subset] = ConcatDataset(datasets)

        if n_samples is not None and n_samples < len(combined_dataset):
            indices = torch.randperm(len(combined_dataset))[:n_samples].tolist()
            combined_dataset = Subset(combined_dataset, indices)

        logger.info(
            f"Evaluating baseline UED for '{augmentation_name}' on {len(combined_dataset)} samples "
            f"from split(s): {'+'.join(splits)} …"
        )
        return self.evaluate_dataset(combined_dataset, batch_size=batch_size, num_workers=num_workers)


# ---------------------------------------------------------------------------
# ABX Evaluator
# ---------------------------------------------------------------------------


class ABXEvaluator:
    """Evaluates ABX phoneme discrimination using the fastabx library.

    Computes Within-Speaker and Across-Speaker ABX error rates following
    the ZeroSpeech 2021 protocol.

    Requirements
    ------------
    - **Python 3.12+** (fastabx uses PEP 695 type aliases, not compatible with 3.11)
    - fastabx installed: ``pip install -e fastabx/``  (submodule) or ``pip install fastabx``
    - Per-utterance feature files saved as PyTorch tensors (``.pt``)
    - A ZeroSpeech-format item file with phoneme alignments

    Item file format (space-separated, with header line starting with ``#``)::

        #file onset offset #phone prev-phone next-phone speaker
        utt_id 0.120 0.340 p      sil        a          spkr1
        utt_id 0.340 0.580 a      p          t          spkr1

    The ``#file`` column must match the stem of the ``.pt`` feature files.

    Usage
    -----
    ::

        evaluator = ABXEvaluator(upstream_encoder=E0.dense, device=device)

        # Step 1 – extract HuBERT features (one .pt per utterance)
        evaluator.extract_features(audio_files, "path/to/features/")

        # Step 2 – run ABX
        scores = evaluator.evaluate("path/to/features/", "librispeech.item")
        # → {"within_speaker_abx": 5.3, "across_speaker_abx": 8.1}
    """

    # ZeroSpeech 2021 default subsampling parameters
    MAX_SIZE_GROUP: int = 10
    MAX_X_ACROSS: int = 5
    FEATURE_FREQUENCY: int = 50  # HuBERT outputs ~50 frames/second

    def __init__(
        self,
        upstream_encoder: torch.nn.Module,
        device: torch.device,
        sample_rate: int = 16000,
    ):
        self.upstream = upstream_encoder
        self.device = device
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_features(
        self,
        audio_files: List[str],
        output_dir: str,
    ) -> str:
        """Extract HuBERT hidden-state features and save as ``.pt`` files.

        Each file is saved as a 2-D float32 tensor of shape
        ``(num_frames, feature_dim)`` — the format expected by fastabx's
        ``Dataset.from_item(feature_maker=torch.load, extension=".pt")``.

        Args:
            audio_files: Paths to audio files (FLAC/WAV).
            output_dir:  Directory where one ``.pt`` file per utterance is written.
                         The filename stem matches the ``#file`` column in the item file.

        Returns:
            ``output_dir`` (for chaining).
        """
        import torchaudio

        os.makedirs(output_dir, exist_ok=True)
        self.upstream.eval()

        for audio_path in audio_files:
            import soundfile as _sf
            data, sr = _sf.read(audio_path, dtype="float32", always_2d=False)
            waveform = torch.from_numpy(data)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(0, keepdim=True)

            waveform = waveform.squeeze(0).unsqueeze(0).to(self.device)  # [1, time]
            feats, _ = self.upstream(waveform, lengths=None)
            # [seq_len, dim] float32 – exactly what fastabx.Dataset.from_item expects
            feat_tensor = feats.squeeze(0).cpu().float()

            file_id = Path(audio_path).stem
            torch.save(feat_tensor, os.path.join(output_dir, f"{file_id}.pt"))
            logger.debug(f"Saved features for {file_id}: {feat_tensor.shape}")

        logger.info(f"Extracted features for {len(audio_files)} files → {output_dir}")
        return output_dir

    # ------------------------------------------------------------------
    def run_abx(
        self,
        features_dir: str,
        item_file: str,
        within_speaker: bool = True,
        distance: str = "angular",
        seed: int = 0,
    ) -> float:
        """Run ABX using ``fastabx.zerospeech_abx`` and return the error rate.

        Internally uses the ZeroSpeech 2021 protocol:
        - context = "within" (triphone context, i.e. by prev/next phone)
        - speaker = "within" or "across"
        - Subsampling: max_size_group=10, max_x_across=5

        Args:
            features_dir:   Directory with per-utterance ``.pt`` feature tensors.
            item_file:      ZeroSpeech-format ``.item`` file with phoneme alignments.
            within_speaker: True → within-speaker; False → across-speaker.
            distance:       "angular" (default, cosine-equivalent), "cosine",
                            or "euclidean".
            seed:           Random seed for subsampling reproducibility.

        Returns:
            ABX error rate (%, lower is better).

        Raises:
            ImportError: If fastabx is not installed or Python < 3.12.
        """
        try:
            from fastabx import zerospeech_abx
        except ImportError as e:
            raise ImportError(
                "fastabx requires Python ≥ 3.12.\n"
                "  1. Update .python-version to 3.12\n"
                "  2. Install: pip install -e fastabx/  (submodule already cloned)\n"
                "  Repository: https://github.com/bootphon/fastabx"
            ) from e

        speaker_mode = "within" if within_speaker else "across"
        score = zerospeech_abx(
            item=item_file,
            root=features_dir,
            max_size_group=self.MAX_SIZE_GROUP,
            max_x_across=self.MAX_X_ACROSS,
            speaker=speaker_mode,
            context="within",        # always evaluate within triphone context
            distance=distance,
            frequency=self.FEATURE_FREQUENCY,
            extension=".pt",
            seed=seed,
        )
        # zerospeech_abx returns a float in [0, 1]; multiply by 100 for %
        return float(score) * 100.0

    def evaluate(
        self,
        features_dir: str,
        item_file: str,
        distance: str = "angular",
        seed: int = 0,
    ) -> Dict[str, float]:
        """Run both within-speaker and across-speaker ABX.

        Args:
            features_dir: Directory with ``.pt`` feature files.
            item_file:    ZeroSpeech-format item file.
            distance:     Distance metric ("angular", "cosine", "euclidean").
            seed:         Subsampling seed.

        Returns:
            Dict with 'within_speaker_abx' and 'across_speaker_abx' (%, lower is better).
        """
        return {
            "within_speaker_abx": self.run_abx(
                features_dir, item_file, within_speaker=True, distance=distance, seed=seed
            ),
            "across_speaker_abx": self.run_abx(
                features_dir, item_file, within_speaker=False, distance=distance, seed=seed
            ),
        }


# ---------------------------------------------------------------------------
# Lexical Evaluator (sWUGGY)
# ---------------------------------------------------------------------------


class LexicalEvaluator:
    """sWUGGY evaluator: LM should prefer real words over pseudo-words.

    Args:
        lm: Object with a ``log_prob(sequence: List[int]) -> float`` method.
            Compatible with :class:`~src.metrics.NgramLM` and any LSTM LM
            with the same interface.
    """

    def __init__(self, lm: NgramLM):
        self.lm = lm

    def score(
        self,
        real_sequences: List[List[int]],
        pseudo_sequences: List[List[int]],
    ) -> Dict[str, float]:
        """Compute sWUGGY accuracy.

        Returns:
            Dict with 'swuggy' (%, higher is better).
        """
        acc = swuggy_score(self.lm.log_prob, real_sequences, pseudo_sequences)
        return {"swuggy": acc}


# ---------------------------------------------------------------------------
# Syntactic Evaluator (sBLIMP)
# ---------------------------------------------------------------------------


class SyntacticEvaluator:
    """sBLIMP evaluator: LM should prefer grammatically correct sentences.

    Args:
        lm: Object with a ``log_prob(sequence: List[int]) -> float`` method.
    """

    def __init__(self, lm: NgramLM):
        self.lm = lm

    def score(
        self,
        good_sequences: List[List[int]],
        bad_sequences: List[List[int]],
    ) -> Dict[str, float]:
        """Compute sBLIMP accuracy.

        Returns:
            Dict with 'sblim' (%, higher is better).
        """
        acc = sblim_score(self.lm.log_prob, good_sequences, bad_sequences)
        return {"sblim": acc}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _paired_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate (clean, augmented) waveform pairs with zero-padding."""
    clean_waves = [b[0].squeeze(0) for b in batch]
    aug_waves = [b[1].squeeze(0) for b in batch]

    max_clean = max(w.shape[-1] for w in clean_waves)
    max_aug = max(w.shape[-1] for w in aug_waves)

    padded_clean = torch.zeros(len(batch), max_clean)
    padded_aug = torch.zeros(len(batch), max_aug)
    clean_lens = torch.zeros(len(batch), dtype=torch.long)
    aug_lens = torch.zeros(len(batch), dtype=torch.long)

    for i, (cw, aw) in enumerate(zip(clean_waves, aug_waves)):
        clean_lens[i] = cw.shape[-1]
        padded_clean[i, : cw.shape[-1]] = cw
        aug_lens[i] = aw.shape[-1]
        padded_aug[i, : aw.shape[-1]] = aw

    return padded_clean, clean_lens, padded_aug, aug_lens


def _build_single_aug_config(
    augmentation_name: str,
    base_cfg: DictConfig,
) -> DictConfig:
    """Return a copy of the augmentation config with only one augmentation active.

    Handles both simple-boolean keys (time_stretch, pitch_shift, reverberation,
    noise) and dict-style keys (echo, random_noise, …) that have an 'enabled'
    sub-key.

    Args:
        augmentation_name: The augmentation to keep enabled.
        base_cfg:          The full augmentations config node from Hydra.

    Returns:
        A new DictConfig with all other augmentations disabled.
    """
    cfg: dict = OmegaConf.to_container(base_cfg, resolve=True)  # type: ignore[assignment]

    # Keys that are plain booleans
    simple_bool_keys = {"time_stretch", "pitch_shift", "reverberation", "noise"}
    # Keys that are sub-dicts with an 'enabled' field
    dict_keys = {
        "echo",
        "random_noise",
        "pink_noise",
        "lowpass_filter",
        "highpass_filter",
        "bandpass_filter",
        "smooth",
        "boost_audio",
        "duck_audio",
        "updownresample",
    }

    for key in simple_bool_keys:
        if key in cfg:
            cfg[key] = key == augmentation_name

    for key in dict_keys:
        if key in cfg and isinstance(cfg[key], dict):
            cfg[key]["enabled"] = key == augmentation_name

    return OmegaConf.create(cfg)
