import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
import random
import os
import glob
import soundfile as sf
import numpy as np
# torchaudio >= 2.5 defaults to torchcodec which requires an extra install.
# Force soundfile backend which works out of the box on all platforms.
try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass  # older versions don't need this call


class AugmentationPipeline:
    def __init__(self, sample_rate=16000, config=None):
        self.sample_rate = sample_rate
        self.config = config or {}
        
        # Cache resamplers for fixed stretch rates to avoid OOM in DataLoader workers
        self.stretch_rates = [0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2]
        self.resamplers = {}
        if self.config.get('time_stretch'):
            for rate in self.stretch_rates:
                new_freq = int(self.sample_rate * rate)
                self.resamplers[rate] = (
                    T.Resample(orig_freq=self.sample_rate, new_freq=new_freq),
                    T.Resample(orig_freq=new_freq, new_freq=self.sample_rate)
                )
                
        # Cache pitch shifters
        self.pitch_shifters = {}
        if self.config.get('pitch_shift'):
            for n in range(-4, 5):
                if n != 0:
                    self.pitch_shifters[n] = T.PitchShift(sample_rate=self.sample_rate, n_steps=n)

    def time_stretch(self, waveform):
        """Apply random time stretching by resampling at a perturbed rate."""
        rate = random.choice(self.stretch_rates)
        
        # Simple resampling to approximate time stretching / speed change
        resample_down, resample_up = self.resamplers[rate]
        stretched = resample_down(waveform)
        
        # Resample back to original rate to fit shapes (pitch and speed change)
        return resample_up(stretched)

    def pitch_shift(self, waveform):
        """Apply random pitch shifting using torchaudio.transforms."""
        n_steps = random.randint(-4, 4)
        if n_steps == 0:
            return waveform
            
        pitch_shifter = self.pitch_shifters[n_steps]
        return pitch_shifter(waveform)

    def add_reverberation(self, waveform):
        """Add synthetic reverberation using a simple echo delay heuristic."""
        # Simple echo delay heuristic since sox reverb is missing
        delay = int(self.sample_rate * 0.05) # 50ms delay
        decay = random.uniform(0.3, 0.8)
        
        echo = torch.zeros_like(waveform)
        if waveform.shape[-1] > delay:
            echo[..., delay:] = waveform[..., :-delay] * decay
            
        return waveform + echo

    def add_noise(self, waveform):
        """Add Gaussian noise at a random SNR between 10 and 40 dB."""
        snr_db = random.uniform(10.0, 40.0)
        signal_power = waveform.norm(p=2)
        noise = torch.randn_like(waveform)
        noise_power = noise.norm(p=2)
        if noise_power == 0:
            return waveform
        snr_linear = 10 ** (snr_db / 20)
        scale = signal_power / (snr_linear * noise_power)
        noisy = waveform + scale * noise
        return noisy

    def __call__(self, waveform):
        if self.config.get('time_stretch'):
            waveform = self.time_stretch(waveform)
        if self.config.get('pitch_shift'):
            waveform = self.pitch_shift(waveform)
        if self.config.get('reverberation'):
            waveform = self.add_reverberation(waveform)
        if self.config.get('noise'):
            waveform = self.add_noise(waveform)
        return waveform


class AudioDataset(Dataset):
    """
    Audio dataset compatible with LibriSpeech-style directory layouts.
    
    Supports two common structures:
      1. Flat: root/<split>/*.flac  (or .wav)
      2. LibriSpeech: root/<split>/<speaker>/<chapter>/<utterance>.flac
    
    Also supports a manifest CSV file at root/<split>.csv with columns:
      file_path, [optional extra columns]
    """

    def __init__(self, root, split="train", augment=False, config=None,
                 target_sr=16000, max_length=None):
        self.root = root
        self.split = split
        self.augment = augment
        self.target_sr = target_sr
        self.max_length = max_length  # max samples; None = no truncation
        self.augmenter = AugmentationPipeline(
            sample_rate=target_sr, config=config
        ) if augment else None
        self.files = self._load_files()

    def _load_files(self):
        """
        Scan for audio files in self.root / self.split.
        
        Tries in order:
          1. CSV manifest at <root>/<split>.csv
          2. Recursive glob for .flac / .wav files under <root>/<split>/
        """
        split_dir = os.path.join(self.root, self.split)

        # Strategy 1: CSV manifest
        csv_path = os.path.join(self.root, f"{self.split}.csv")
        if os.path.isfile(csv_path):
            import csv
            files = []
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Accept common column names
                    path = row.get("file_path") or row.get("path") or row.get("audio")
                    if path:
                        # Resolve relative paths against root
                        if not os.path.isabs(path):
                            path = os.path.join(self.root, path)
                        files.append(path)
            if files:
                return files

        # Strategy 2: Recursive glob
        if os.path.isdir(split_dir):
            extensions = ("**/*.flac", "**/*.wav", "**/*.mp3", "**/*.ogg")
            files = []
            for ext in extensions:
                files.extend(
                    glob.glob(os.path.join(split_dir, ext), recursive=True)
                )
            files.sort()  # deterministic ordering
            return files

        # Fallback: try root itself (flat layout, no split subdirectory)
        if os.path.isdir(self.root):
            extensions = ("**/*.flac", "**/*.wav", "**/*.mp3", "**/*.ogg")
            files = []
            for ext in extensions:
                files.extend(
                    glob.glob(os.path.join(self.root, ext), recursive=True)
                )
            files.sort()
            return files

        return []

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        # backend="soundfile" bypasses torchcodec (not supported on Windows)
        data, sample_rate = sf.read(file_path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)

        # Convert to mono if multi-channel
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if necessary (using functional to avoid caching infinite rates, usually it's just 16k)
        if sample_rate != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.target_sr)

        # Truncate to max_length if set
        if self.max_length is not None and waveform.shape[-1] > self.max_length:
            start = random.randint(0, waveform.shape[-1] - self.max_length)
            waveform = waveform[:, start:start + self.max_length]

        clean_waveform = waveform.clone()

        if self.augment and self.augmenter:
            with torch.no_grad():
                augmented_waveform = self.augmenter(waveform)
            return clean_waveform.detach(), augmented_waveform.detach()

        return clean_waveform.detach()