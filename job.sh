#!/bin/bash
#SBATCH --job-name=500_no_extra_augs
#SBATCH --output=outputs/500_no_extra_augs_slurm_%j.out
#SBATCH --error=outputs/500_no_extra_augs_slurm_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00

uv run python train.py training.run_name="500_no_extra_augs" model.vocab_size=500 dataset.augmentations.max_augs=1 dataset.augmentations.activate_extra_augs=False 

# training.resume_from="/home/infres/lasri-22/snlp_project/outputs/500_long/2026-03-21/13-42-49/round_0/E1_last.pt"
