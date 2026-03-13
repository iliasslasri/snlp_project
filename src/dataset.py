import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
import random
import os
import glob
import numpy as np
import julius
from julius import fft_conv1d, resample_frac
import pyroomacoustics as pra


def generate_pink_noise(length: int) -> torch.Tensor:
    """Generate pink noise using Voss-McCartney algorithm."""
    num_rows = 16
    array = torch.randn(num_rows, length // num_rows + 1)
    reshaped_array = torch.cumsum(array, dim=1)
    reshaped_array = reshaped_array.reshape(-1)[:length]
    return reshaped_array / (torch.max(torch.abs(reshaped_array)) + 1e-8)


class AugmentationPipeline:
    def __init__(self, sample_rate=16000, config=None, noise_dir=None):
        self.sample_rate = sample_rate
        self.config = config or {}
        self.max_augs = self.config.get('max_augs', 4)  # number of augmentations to apply per sample
                                                # we sample k ~ Uniform(0, n_aug) augmentations to apply per sample

        # Scan noise directory for background noise files (DNS challenge)
        self.noise_files = []
        if noise_dir and os.path.isdir(noise_dir):
            for ext in ("*.wav", "*.flac", "*.mp3", "*.ogg"):
                self.noise_files.extend(
                    glob.glob(os.path.join(noise_dir, "**", ext), recursive=True)
                )
            self.noise_files.sort()
            
        # Scan rir directory for offline room impulse responses
        self.rir_files = []
        rir_dir = self.config.get('rir_dir')
        if rir_dir and os.path.isdir(rir_dir):
            for ext in ("*.wav", "*.flac", "*.mp3", "*.ogg"):
                self.rir_files.extend(
                    glob.glob(os.path.join(rir_dir, "**", ext), recursive=True)
                )
            self.rir_files.sort()
        
        # Cache resamplers for fixed stretch rates
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

        self.available = []
        if self.config.get('reverberation'):
            self.available.append(self.add_reverberation)
        if self.config.get('noise'):
            self.available.append(self.add_noise)
        if self.config.get('time_stretch'):
            self.available.append(self.time_stretch)
        if self.config.get('pitch_shift'):
            self.available.append(self.pitch_shift)
        # new augmentations
        if self._aug_enabled('echo'):
            self.available.append(self.echo)
        if self._aug_enabled('random_noise'):
            self.available.append(self.random_noise)
        if self._aug_enabled('pink_noise'):
            self.available.append(self.pink_noise_aug)
        if self._aug_enabled('lowpass_filter'):
            self.available.append(self.lowpass_filter)
        if self._aug_enabled('highpass_filter'):
            self.available.append(self.highpass_filter)
        if self._aug_enabled('bandpass_filter'):
            self.available.append(self.bandpass_filter)
        if self._aug_enabled('smooth'):
            self.available.append(self.smooth)
        if self._aug_enabled('boost_audio'):
            self.available.append(self.boost_audio)
        if self._aug_enabled('duck_audio'):
            self.available.append(self.duck_audio)
        if self._aug_enabled('updownresample'):
            self.available.append(self.updownresample)

    def _aug_enabled(self, name):
        """Check if an augmentation sub-config exists and has enabled=true."""
        cfg = self.config.get(name)
        if cfg is None:
            return False
        if isinstance(cfg, dict):
            return cfg.get('enabled', False)
        # DictConfig from omegaconf
        return getattr(cfg, 'enabled', False)
            
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
        """Apply reverberation by convolving with a precomputed or simulated Room Impulse Response (RIR)."""
        orig_len = waveform.shape[-1]
        
        # Fast path: use offline RIRs
        if self.rir_files:
            rir_path = random.choice(self.rir_files)
            rir, sr = torchaudio.load(rir_path)
            
            # Resample RIR if necessary
            if sr != self.sample_rate:
                rir = torchaudio.functional.resample(rir, sr, self.sample_rate)
                
            rir = rir.to(waveform.device)
            
            # Apply convolution using torchaudio.functional.fftconvolve (fast)
            # Both waveform and rir should be [channel, time].
            reverbed = torchaudio.functional.fftconvolve(waveform, rir, mode="full")
            
            # Crop back to original length
            reverbed = reverbed[..., :orig_len]
            
            # Normalize to preserve original energy
            orig_rms = waveform.norm(p=2) + 1e-8
            reverbed_rms = reverbed.norm(p=2) + 1e-8
            reverbed = reverbed * (orig_rms / reverbed_rms)
            
            return reverbed

        # Slow fallback path: simulate on-the-fly (useful for debugging, but very slow)
        # Random room dimensions (meters)
        room_x = random.uniform(3.0, 8.0)
        room_y = random.uniform(3.0, 6.0)
        room_z = random.uniform(2.5, 4.0)

        # Random RT60 reverberation time (seconds)
        rt60 = random.uniform(0.2, 0.8)

        # Compute absorption and max_order from RT60 using Sabine's formula
        try:
            e_absorption, max_order = pra.inverse_sabine(rt60, [room_x, room_y, room_z])
        except ValueError:
             e_absorption = 0.2
             max_order = 10

        room = pra.ShoeBox(
            [room_x, room_y, room_z],
            fs=self.sample_rate,
            materials=pra.Material(e_absorption),
            max_order=max_order,
        )

        # Random source position (inside the room with margin)
        margin = 0.3
        src_pos = [
            random.uniform(margin, room_x - margin),
            random.uniform(margin, room_y - margin),
            random.uniform(margin, room_z - margin),
        ]

        # Random microphone position (inside the room with margin)
        mic_pos = [
            random.uniform(margin, room_x - margin),
            random.uniform(margin, room_y - margin),
            random.uniform(margin, room_z - margin),
        ]

        # Convert waveform to numpy for pyroomacoustics
        signal_np = waveform.squeeze(0).cpu().numpy()

        room.add_source(src_pos, signal=signal_np)
        room.add_microphone(mic_pos)
        room.simulate()

        reverbed = room.mic_array.signals[0]  # shape: (n_samples,)

        # Crop back to original length
        reverbed = reverbed[:orig_len]

        # Normalize to preserve original energy
        orig_rms = np.sqrt(np.mean(signal_np ** 2)) + 1e-8
        reverbed_rms = np.sqrt(np.mean(reverbed ** 2)) + 1e-8
        reverbed = reverbed * (orig_rms / reverbed_rms)

        result = torch.from_numpy(reverbed.astype(np.float32)).unsqueeze(0)
        return result.to(waveform.device)

    def _load_noise(self, target_length):
        """Load a random noise clip, loop/crop to match target_length samples."""
        noise_path = random.choice(self.noise_files)
        noise, sr = torchaudio.load(noise_path)
        # Convert to mono
        if noise.shape[0] > 1:
            noise = noise.mean(dim=0, keepdim=True)
        # Resample to target sample rate
        if sr != self.sample_rate:
            noise = torchaudio.functional.resample(noise, sr, self.sample_rate)
        # Loop if noise is shorter than signal
        if noise.shape[-1] < target_length:
            repeats = (target_length // noise.shape[-1]) + 1
            noise = noise.repeat(1, repeats)
        # Crop to exact length
        noise = noise[:, :target_length]
        return noise

    def add_noise(self, waveform):
        """Mix with real background noise (DNS challenge) at SNR in [5, 15] dB.
        Falls back to Gaussian noise if no noise files are available."""
        snr_db = random.uniform(5.0, 15.0)

        if self.noise_files:
            noise = self._load_noise(waveform.shape[-1]).to(waveform.device)
        else:
            noise = torch.randn_like(waveform)

        signal_power = waveform.norm(p=2)
        noise_power = noise.norm(p=2)
        if noise_power == 0:
            return waveform
        snr_linear = 10 ** (snr_db / 20)
        scale = signal_power / (snr_linear * noise_power)
        return waveform + scale * noise

    def _aug_cfg(self, name):
        """Return the sub-config dict for an augmentation."""
        cfg = self.config.get(name, {})
        if isinstance(cfg, dict):
            return cfg
        # OmegaConf DictConfig: convert to plain dict
        try:
            from omegaconf import OmegaConf
            return OmegaConf.to_container(cfg, resolve=True)
        except ImportError:
            return dict(cfg)

    def echo(self, waveform):
        """Add echo by convolving with a simple impulse response."""
        cfg = self._aug_cfg('echo')
        volume_range = cfg.get('volume_range', [0.1, 0.5])
        duration_range = cfg.get('duration_range', [0.1, 0.5])

        duration = random.uniform(*duration_range)
        volume = random.uniform(*volume_range)
        n_samples = int(self.sample_rate * duration)
        if n_samples < 2:
            return waveform

        ir = torch.zeros(n_samples, dtype=waveform.dtype, device=waveform.device)
        ir[0] = 1.0
        ir[-1] = volume
        ir = ir.unsqueeze(0).unsqueeze(0)  # [1, 1, time]

        # waveform is [1, time] -> add batch dim for fft_conv1d
        out = fft_conv1d(waveform.unsqueeze(0), ir).squeeze(0)
        # Preserve amplitude
        max_orig = torch.max(torch.abs(waveform)) + 1e-8
        max_out = torch.max(torch.abs(out)) + 1e-8
        out = out / max_out * max_orig
        # Crop to original length
        return out[..., :waveform.shape[-1]]

    def random_noise(self, waveform):
        """Add Gaussian noise."""
        cfg = self._aug_cfg('random_noise')
        noise_std = cfg.get('noise_std', 0.001)
        return waveform + torch.randn_like(waveform) * noise_std

    def pink_noise_aug(self, waveform):
        """Add pink background noise."""
        cfg = self._aug_cfg('pink_noise')
        noise_std = cfg.get('noise_std', 0.01)
        noise = generate_pink_noise(waveform.shape[-1]) * noise_std
        return waveform + noise.unsqueeze(0).to(waveform.device)

    def lowpass_filter(self, waveform):
        """Apply lowpass filter."""
        cfg = self._aug_cfg('lowpass_filter')
        cutoff = cfg.get('cutoff_freq', 5000)
        return julius.lowpass_filter(waveform, cutoff=cutoff / self.sample_rate)

    def highpass_filter(self, waveform):
        """Apply highpass filter."""
        cfg = self._aug_cfg('highpass_filter')
        cutoff = cfg.get('cutoff_freq', 500)
        return julius.highpass_filter(waveform, cutoff=cutoff / self.sample_rate)

    def bandpass_filter(self, waveform):
        """Apply bandpass filter."""
        cfg = self._aug_cfg('bandpass_filter')
        low = cfg.get('cutoff_freq_low', 300)
        high = cfg.get('cutoff_freq_high', 8000)
        return julius.bandpass_filter(
            waveform,
            cutoff_low=low / self.sample_rate,
            cutoff_high=high / self.sample_rate,
        )

    def smooth(self, waveform):
        """Smooth via moving-average convolution."""
        cfg = self._aug_cfg('smooth')
        wr = cfg.get('window_size_range', [2, 10])
        window_size = random.randint(int(wr[0]), int(wr[1]))
        kernel = torch.ones(1, 1, window_size, dtype=waveform.dtype,
                            device=waveform.device) / window_size
        out = fft_conv1d(waveform.unsqueeze(0), kernel).squeeze(0)
        # Pad to original length
        result = torch.zeros_like(waveform)
        result[..., :out.shape[-1]] = out
        return result

    def boost_audio(self, waveform):
        """Amplify signal by a percentage."""
        cfg = self._aug_cfg('boost_audio')
        amount = cfg.get('amount', 20)
        return waveform * (1 + amount / 100)

    def duck_audio(self, waveform):
        """Attenuate signal by a percentage."""
        cfg = self._aug_cfg('duck_audio')
        amount = cfg.get('amount', 20)
        return waveform * (1 - amount / 100)

    def updownresample(self, waveform):
        """Upsample then downsample to introduce resampling artifacts."""
        cfg = self._aug_cfg('updownresample')
        intermediate = cfg.get('intermediate_freq', 32000)
        up = resample_frac(waveform, self.sample_rate, intermediate)
        return resample_frac(up, intermediate, self.sample_rate)[..., :waveform.shape[-1]]

    def __call__(self, waveform):
        if self.available:
            # Randomly sample 0 to min(4, len(available)) augmentations
            k = random.randint(0, min(self.max_augs, len(self.available)))
            selected = random.sample(self.available, k)
            for aug_fn in selected:
                waveform = aug_fn(waveform)

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
                 target_sr=16000, max_length=None, noise_dir=None):
        self.root = root
        self.split = split
        self.augment = augment
        self.target_sr = target_sr
        self.max_length = max_length  # max samples; None = no truncation
        self.augmenter = AugmentationPipeline(
            sample_rate=target_sr, config=config, noise_dir=noise_dir
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
        waveform, sample_rate = torchaudio.load(file_path)

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
