#!/bin/bash
#SBATCH -t 3-08:00:00
#SBATCH --mem=40G
#SBATCH --array=0-50
source /data/Ian_Data/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
python3 FetchReach_SAC_HPC_Sweep.py $SLURM_ARRAY_TASK_ID