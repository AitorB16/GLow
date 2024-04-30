#!/bin/bash

#SBATCH --job-name=FGL

#SBATCH --output=logs/M_%A_%a.out
#SBATCH --error=logs/M_%A_%a.err

#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
# #SBATCH --ntasks-per-core=1

# #SBATCH --mem-per-cpu=2G
#SBATCH --mem-per-cpu=8G

#SBATCH --partition=GPU
#SBATCH --gpus=1

#SBATCH --array=1-1 #1 job to 1 GPU

export PYTHONPATH=.venv/lib/python3.10/site-packages/:../libs/
#export APPTAINERENV_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
# bnd -exec echo $LD_LIBRARY_PATH

bnd -exec python3  "$@" #> ./out.txt
