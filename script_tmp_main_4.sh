#!/bin/bash

#SBATCH --time=6:00:00
#SBATCH --job-name=tmp_main_ssl_4
#SBATCH --account=PAS2622

#SBATCH --mem=32gb

#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH -o /fs/scratch/PAS2622/ssl_based/op_script_tmp_main_4.out

module load miniconda3

module load cuda/11.8.0
source activate local
echo $CONDA_DEFAULT_ENV
python /fs/scratch/PAS2622/ssl_based/tmp_main.py