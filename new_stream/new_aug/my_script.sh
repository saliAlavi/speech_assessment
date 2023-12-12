#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --job-name=new_aug
#SBATCH --account=PAS2622
#SBATCH -o /fs/scratch/PAS2622/ssl_based/new_stream/new_aug/new_aug_regression.out

#SBATCH --mem=32gb

#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2

module load miniconda3

module load cuda/11.8.0
source activate local
echo $CONDA_DEFAULT_ENV

python /fs/scratch/PAS2622/ssl_based/new_stream/new_aug/new_aug_regression.py
