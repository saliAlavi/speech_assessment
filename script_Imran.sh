#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --job-name=ds_nisqa_create
#SBATCH --account=PAS2622

#SBATCH --mem=32gb

#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1

module load miniconda3
module load cuda
source activate local
python main_Imran.py
