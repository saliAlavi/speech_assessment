#!/bin/bash

#SBATCH --time=15:00:00 
#SBATCH --job-name=ds_insqa_create
#SBATCH --account=PAS2622
#SBATCH --mem=32gb

#SBATCH --cpus-per-task=8
#SBATCH -o /fs/scratch/PAS2622/ssl_based/create_ds/nisqa/sbatch_output.out

module load miniconda3
source activate /users/PAS2301/alialavi/miniconda3/envs/tf
cd /fs/scratch/PAS2622/ssl_based/create_ds/nisqa 
tfds build --overwrite
