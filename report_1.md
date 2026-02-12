

## Project Objectives

1. Reproduce & Benchmark
    - Evaluate the robustness of baseline discrete units (HuBERT, Wav2Vec 2.0) using the ABX metric and the lev distance.
    - Implement the augmentations from the paper and reproduce the results.

2. Implementation of robust quantization
    - Pseudo-labeling: should be implemented.
    - Training robust quantizers

3. Extensions
    - Suggested in [projets proposal](https://github.com/rbawden/MVA_2026_SNLP/blob/main/projects.md):
      - Adding more augmentations.
      - try modern encoders
      - Cross-lingual Robustness: Testing if the robust quantizer trained on English generalizes to phonemes in other languages
    - Suggested by us:
      - Adapt the evaluation to more used encoders like Encodec [Arxiv paper](https://arxiv.org/pdf/2210.13438), [huggingface](https://huggingface.co/facebook/encodec_32khz)