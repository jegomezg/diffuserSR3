#!/bin/bash -l
#SBATCH --chdir=/share/projects/ottopia/superstable/SRdiff
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 16
#SBATCH --partition ottopia
#SBATCH --output test.out
#SBATCH --time=2-00:00:00


conda activate superstable

python3 t.py -p test -c config.json 
