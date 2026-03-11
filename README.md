# snlp_project

## Environment Setup

```bash
uv sync        # install all deps
uv sync --dev  # include dev tools (ruff, ipykernel)
```

To download the dataset, run:
```bash
bash src/scripts/download_dataset.sh
```
and
```bash
bash src/scripts/download-dns-challenge-5-noise-ir.sh
```

run a training:

First, generate RIRs:
```bash
uv run python src/scripts/simulate_rirs.py --n 50 # in our training we generate 4000 RIRs
```

then run the training:
```bash
uv run python train.py

```

run tensorboard:
```bash
uv run tensorboard --logdir checkpoints/quantizer/runs --port 6006
```

or 
```bash
source .venv/bin/activate
tensorboard --logdir checkpoints/quantizer/runs --port 6006
```
