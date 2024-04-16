#!/bin/bash

#SBATCH --job-name=FGL

#SBATCH --output=logs/M_%A_%a.out
#SBATCH --error=logs/M_%A_%a.err

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --threads-per-core=1
# #SBATCH --ntasks-per-core=2

# #SBATCH --mem-per-cpu=2G
#SBATCH --mem-per-cpu=8G

#SBATCH --partition=GPU
#SBATCH --gpus=2

#SBATCH --array=1-1 #1 job to 1 GPU

export PYTHONPATH=../../flower/simulation/.venv/lib/python3.10/site-packages/:../libs/
#export APPTAINERENV_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
# bnd -exec echo $LD_LIBRARY_PATH

bnd -exec python3  "$@" #> ./out.txt
