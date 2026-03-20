# Evaluation — GSLM Robustness Model

This document describes the evaluation pipeline for the augmentation-invariant
discrete representation model (E1), reproducing the tables from
*Augmentation Invariant Discrete Representation for Generative Spoken Language Modeling*.

---

## Installation

### 1. Python 3.12

fastabx uses PEP 695 syntax (`type X = ...`) which requires Python 3.12.
The `.python-version` file is already set to `3.12`. Install it and recreate the
virtual environment:

```bash
uv python install 3.12
uv sync              # base deps (torch ≥ 2.1.2, torchaudio, transformers, …)
```

### 2. Download LibriSpeech

```bash
uv run --with datasets python scripts/download_librispeech.py --split test-clean
```

Available splits: `train-clean-100`, `train-clean-360`, `train-other-500`, `dev-clean`, `dev-other`, `test-clean`, `test-other`.

### 3. ABX dependencies

fastabx is declared as a local editable dependency in the `eval` group,
pointing to the `fastabx/` submodule:

```bash
# Clone the submodule if not already done:
git submodule update --init fastabx

# Install fastabx + its deps (polars, torchdtw, …)
uv sync --group eval
```

> **Note on torch version:** fastabx requires `torch>=2.10.0`.
> `uv sync --group eval` will upgrade torch automatically if needed.
> Base training deps are unpinned (`torch>=2.1.2`) so there is no conflict.

### Quick check

```bash
uv run python -c "import fastabx; print('fastabx OK')"
```

---

## File overview

| File | Role |
|---|---|
| `src/metrics.py` | Low-level metric functions (UED, Levenshtein, sWUGGY, sBLIMP, NgramLM) |
| `src/evaluation.py` | High-level evaluator classes (UEDEvaluator, ABXEvaluator, …) |
| `evaluate.py` | CLI script — run any subset of metrics on a checkpoint |
| `scripts/augmentation_report.py` | Per-augmentation UED table (single or multiple checkpoints) |
| `scripts/ablation_report.py` | UED table across vocabulary sizes (50/100/200/500) |

---

## Metrics

### Unit Edit Distance (UED) — robustness metric

Defined in the paper. Measures how much the discrete representation changes
under audio augmentation.

**Algorithm (per sample pair):**
1. Run E1 on clean audio → unit sequence `z_clean`
2. Run E1 on augmented audio → unit sequence `z_aug`
3. Filter blank tokens, then deduplicate both sequences (remove consecutive repeats, e.g. `1 1 2 2 1 → 1 2 1`)
4. Compute Levenshtein edit distance between the deduplicated sequences
5. Divide by `len(z_clean_dedup)` and multiply by 100

Lower is better. 0 = perfectly invariant to augmentation.

**Dependencies:** `editdistance` (C implementation, ~20× faster than pure Python).
Falls back to a pure-Python two-row DP if not installed.

> **DNS noise:** The `noise` augmentation requires real DNS challenge noise files
> (non-stationary). Pass `--noise-dir <path>` to point to your DNS corpus.
> Without it, the code falls back to Gaussian white noise and results will not
> match the paper's numbers for that row.

---

### ABX — phoneme discrimination

Measures whether the model's representations are phonemically discriminative.
Uses the **fastabx** library (ZeroSpeech 2021 protocol).

> **Requires Python 3.12+** and `torch>=2.10.0`. See the [Installation](#installation)
> section above.

**Setup** (one-time, after `uv sync`):
```bash
uv sync --group eval   # installs fastabx from fastabx/ submodule
```

**What it needs:**
- A `.item` file with phoneme alignments in ZeroSpeech format (see below)
- One `.pt` file per utterance containing the HuBERT feature tensor `[frames, 768]`

**Item file format** (space-separated, header starts with `#`):
```
#file onset offset #phone prev-phone next-phone speaker
utt_id 0.120 0.340 p      sil        a          spkr1
utt_id 0.340 0.580 a      p          t          spkr1
```
The `#file` column must match the stem of the `.pt` feature files.

**Two modes:**
- `within_speaker_abx` — A and B from the same speaker
- `across_speaker_abx` — A and B from different speakers

Lower is better. ~5% = excellent, ~10–15% = typical.

**Distance metric:** `angular` (default, ZeroSpeech standard). Equivalent to cosine
similarity up to a monotone transformation; numerically more stable.

---

### sWUGGY — lexical evaluation

Tests whether a language model (LM) trained on discrete units prefers real words
over phonetically similar pseudo-words.

**Input:** JSON file, list of pairs:
```json
[
  {"real": [12, 45, 3, 200], "pseudo": [12, 45, 7, 200]},
  ...
]
```
Each sequence is a list of unit indices (output of E1, already deduplicated).

**Score:** percentage of pairs where `log_prob(real) > log_prob(pseudo)`.
Higher is better. 50% = random.

---

### sBLIMP — syntactic evaluation

Tests whether an LM prefers grammatically correct sentences over incorrect ones.

**Input:** JSON file, list of pairs:
```json
[
  {"good": [5, 12, 3, 44, 8], "bad": [5, 12, 44, 3, 8]},
  ...
]
```

**Score:** percentage of pairs where `log_prob(good) > log_prob(bad)`.
Higher is better.

**LM:** Use `NgramLM` (built-in n-gram LM) or any object with a
`log_prob(sequence: list[int]) -> float` method.

---

## evaluate.py — running individual metrics

```
uv run python evaluate.py --metrics <metric1> [<metric2> ...] [options]
```

Available metrics: `ued`, `abx`, `swuggy`, `sblim`. Default: all four.

### UED only

```bash
uv run python evaluate.py --metrics ued --checkpoint checkpoints/quantizer/E1_best.pt --data-root LibriSpeech --split test-clean --noise-dir noise_fullband --n-samples 500
```

`--n-samples 0` runs on the full split. `--noise-dir` is required to match paper noise results.

### UED with E0 baseline row

```bash
uv run python evaluate.py --metrics ued --checkpoint checkpoints/quantizer/E1_best.pt --data-root LibriSpeech --split test-clean --noise-dir noise_fullband --n-samples 500 --include-baseline
```

Adds an E0 (HuBERT + KMeans, no E1) reference row to the output table.

### ABX only (features already extracted)

```bash
uv run python evaluate.py --metrics abx --abx-items data/librispeech-clean.item --abx-features outputs/features/test-clean
```

No `--checkpoint` needed — E1 is not loaded for ABX.

### ABX with on-the-fly feature extraction

```bash
uv run python evaluate.py --metrics abx --abx-items data/librispeech-clean.item --abx-extract --abx-audio-dir LibriSpeech/test-clean --abx-features outputs/features/test-clean --abx-distance angular
```

One `.pt` file is saved per utterance in `--abx-features`. Re-running without `--abx-extract` reuses the cached files.

### sWUGGY only

```bash
uv run python evaluate.py --metrics swuggy --lm-checkpoint checkpoints/lm/ngram.pkl --swuggy-data data/swuggy_units.json
```

### sBLIMP only

```bash
uv run python evaluate.py --metrics sblim --lm-checkpoint checkpoints/lm/ngram.pkl --sblim-data data/sblim_units.json
```

### UED + ABX together

```bash
uv run python evaluate.py --metrics ued abx --checkpoint checkpoints/quantizer/E1_best.pt --data-root LibriSpeech --split test-clean --noise-dir noise_fullband --n-samples 500 --abx-items data/librispeech-clean.item --abx-features outputs/features/test-clean --output results/eval.json
```

### Full evaluation (all metrics)

```bash
uv run python evaluate.py --checkpoint checkpoints/quantizer/E1_best.pt --data-root LibriSpeech --split test-clean --noise-dir noise_fullband --n-samples 500 --abx-items data/librispeech-clean.item --abx-features outputs/features/test-clean --lm-checkpoint checkpoints/lm/ngram.pkl --swuggy-data data/swuggy_units.json --sblim-data data/sblim_units.json --output results/eval_full.json
```

---

## scripts/augmentation_report.py — per-augmentation comparison table

Evaluates UED for each of the four paper augmentations
(Time Stretch, Pitch Shift, Reverberation, Noise) and prints a formatted table.

### Single checkpoint

```bash
uv run python scripts/augmentation_report.py --checkpoint checkpoints/quantizer/E1_best.pt --data-root LibriSpeech --split test-clean --noise-dir noise_fullband --n-samples 500
```

### Compare multiple checkpoints (iterative rounds)

```bash
uv run python scripts/augmentation_report.py --checkpoints checkpoints/quantizer/round_0/E1_best.pt checkpoints/quantizer/round_1/E1_best.pt checkpoints/quantizer/round_2/E1_best.pt --labels "Round 0" "Round 1" "Round 2" --include-baseline --data-root LibriSpeech --split test-clean --noise-dir noise_fullband --n-samples 500 --markdown
```

`--include-baseline` adds an E0 (HuBERT + KMeans, no E1) row for reference.
`--markdown` prints a Markdown table instead of ASCII. `--output-json` saves raw numbers.

### Example output (ASCII)

```
  UED Comparison Table (lower is better)

                      Baseline (E0)  Round 0  Round 1  Round 2
  ---------------------------------------------------------------
  Time Stretch         22.10±9.80    15.32±8.45    12.10±7.20    10.80±6.50
  Pitch Shift          18.50±8.20    12.18±6.23     9.90±5.80     8.40±5.10
  Reverberation        25.30±10.10   18.75±9.12    14.20±8.00    12.50±7.20
  Noise                20.80±9.00    14.50±7.89    11.30±6.70     9.80±6.00
  ---------------------------------------------------------------
```

---

## scripts/ablation_report.py — vocabulary size ablation table

Reproduces the quantizer ablation table from the paper (UED across K = 50/100/200/500 clusters).
Requires one checkpoint trained per vocabulary size.

### Usage

```bash
uv run python scripts/ablation_report.py --checkpoints checkpoints/quantizer/E1_50.pt checkpoints/quantizer/E1_100.pt checkpoints/quantizer/E1_200.pt checkpoints/quantizer/E1_500.pt --vocab-sizes 50 100 200 500 --data-root LibriSpeech --split test-clean --noise-dir noise_fullband --n-samples 500 --markdown
```

### Example output (Markdown)

```
| Augmentation  | K=50          | K=100         | K=200         | K=500         |
|---|---|---|---|---|
| Time Stretch  | 18.20±9.10    | 15.30±8.40    | 13.10±7.50    | 10.80±6.50    |
| Pitch Shift   | 14.50±7.20    | 12.10±6.50    | 10.20±5.90    |  8.40±5.10    |
| Reverberation | 22.00±10.30   | 19.10±9.50    | 16.30±8.40    | 12.50±7.20    |
| Noise         | 17.80±8.50    | 14.90±7.80    | 12.50±6.90    |  9.80±6.00    |
```

`--output-json` saves raw numbers. Omit `--markdown` for ASCII output.

---

## Paper alignment notes

| Checklist item | Status | Notes |
|---|---|---|
| UED formula `(1/T'_x) * LEV(...)` | ✅ | Blank filtered before dedup; divide by clean length |
| Time Stretch rate [0.8, 1.2] | ✅ | Discrete grid {0.8, 0.85, …, 1.2} |
| Time Stretch method | ⚠️ | Speed perturbation (resample ×2) — model trained with this; not Phase Vocoder |
| Pitch Shift ±4 semitones | ✅ | `torchaudio.transforms.PitchShift` |
| Reverberation pyroomacoustics | ✅ | Randomized room/RT60/mic/source; offline RIRs supported |
| Noise SNR [5, 15] dB | ✅ | `uniform(5, 15)` |
| Noise non-stationary (DNS) | ⚠️ | Requires `--noise-dir`; falls back to Gaussian otherwise |
| ABX within + across speaker | ✅ | Both modes run by default |
| Quantizer ablations 50/100/200/500 | ✅ | `scripts/ablation_report.py` |

---

## Key changes vs initial implementation

| What | Before | After |
|---|---|---|
| `evaluate.py` metric selection | Always ran all metrics | `--metrics ued abx swuggy sblim` — run any subset |
| `--checkpoint` | Required | Optional — not needed for ABX-only runs |
| ABX feature format | `.npy` (numpy) | `.pt` (PyTorch tensor, required by fastabx) |
| ABX API | Fictitious `fastabx.Features`, `fastabx.score()` | Real API: `zerospeech_abx()` |
| ABX distance default | `"cosine"` | `"angular"` (ZeroSpeech 2021 standard) |
| Levenshtein | Pure Python DP | `editdistance` (C, ~20× faster) with Python fallback |
| Python version | 3.11 (`requires-python = ">=3.11"`) | 3.12 — required by fastabx (PEP 695 syntax) |
| torch version | `==2.1.2` (hard pin) | `>=2.1.2` — allows uv to satisfy fastabx's `torch>=2.10.0` |
| fastabx install | Manual `uv pip install -e fastabx/` | `uv sync --group eval` — declared in `pyproject.toml` |
| `transformers` import in `models.py` | Unused `Wav2Vec2Model, HubertModel` imported | Removed — eliminated the "PyTorch ≥ 2.4 required" warning |
| Blank token dedup order | `unique_consecutive` before blank filter | Blank filtered first — avoids `[1, blank, 1] → [1, 1]` edge case |
| E0 baseline | Not available | `--include-baseline` adds HuBERT+KMeans reference row |
| Vocab-size ablation | No script | `scripts/ablation_report.py` |
| Noise DNS warning | Silent fallback | Explicit warning when `--noise-dir` is absent |

---

## Dependencies

| Package | Purpose | Notes |
|---|---|---|
| `editdistance` | Fast Levenshtein (UED) | C implementation; pure-Python fallback if absent |
| `fastabx` | ABX scoring | Requires Python 3.12+; install from submodule |
| `polars` | Internal to fastabx | Installed automatically with fastabx |
| `torchdtw` | Internal to fastabx | Installed automatically with fastabx |
| `pyroomacoustics` | Reverberation simulation | On-the-fly RIR generation |
| `soundfile` | Audio I/O | Replaces `torchaudio.load` (avoids TorchCodec/FFmpeg on Windows) |

tmux attach -t eval