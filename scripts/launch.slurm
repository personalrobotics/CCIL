#!/bin/bash

# Run with sbatch parallel_srun.slurm

#SBATCH --job-name=ccilhover
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-a40
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kayke@uw.edu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate torch

cat $0
echo -e "\n---------------\n"
echo "args"
echo $1
echo -e "\n\n================\n\n"

export TQDM_DISABLE=1

cat $HOME/CorrectiveIL/scripts/sweep.sh
bash $HOME/CorrectiveIL/scripts/sweep.sh
