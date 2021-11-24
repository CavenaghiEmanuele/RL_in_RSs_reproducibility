#!/bin/sh
#SBATCH --time=1

srun python train_transe_model.py --dataset beauty
