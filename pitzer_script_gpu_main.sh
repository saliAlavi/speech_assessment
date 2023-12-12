#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --job-name=ds_nisqa_create
#SBATCH --account=PAS2622

#SBATCH --mem=32gb

#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH -o /fs/scratch/PAS2622/ssl_based/op_script_gpu_main.out

module load miniconda3
module load cuda/11.8.0
source activate /users/PAS2301/alialavi/miniconda3/envs/tf
#conda activate tf
cd /fs/scratch/PAS2622/ssl_based
echo $CONDA_DEFAULT_ENV
pip list
python scratch_file.py
