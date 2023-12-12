#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --job-name=ds_nisqa_create
#SBATCH --account=PAS2301

#SBATCH --mem=32gb

#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH -o /fs/scratch/PAS2622/ssl_based/tmp_main.out

module load miniconda3/23.3.1-py310
module load cuda/11.8.0
#module load cudnn/8.6.0.163-11.8
conda activate tf
conda env list
which python
#cd /fs/scratch/PAS2622/ssl_based
echo $CONDA_DEFAULT_ENV
/users/PAS2301/alialavi/.conda/envs/tf/bin/python tmp_main.py
