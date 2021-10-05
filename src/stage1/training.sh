#!/bin/bash
#SBATCH --job-name=WEATHER
#SBATCH --ntasks=1 --nodes=1
#SBATCH --partition=xgpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --output=slurm-%j-%x.out

module load anaconda/3
module load cuda/10.1

source /home/dinardo/.bashrc
conda activate tf2

python -u stage1.py --settings stage1.yaml
