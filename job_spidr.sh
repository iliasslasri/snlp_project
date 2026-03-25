#!/bin/bash
#SBATCH --job-name=256
#SBATCH --output=outputs/256_slurm_%j.out
#SBATCH --error=outputs/256_slurm_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00

uv run python train.py training.run_name="256" model.name="spidr_base" model.vocab_size=256 dataset.augmentations.max_augs=4 dataset.augmentations.activate_extra_augs=True