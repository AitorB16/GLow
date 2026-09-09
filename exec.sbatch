#!/bin/bash

#SBATCH --job-name=GLow_C_C

#SBATCH --output=logs/M_%A_%a.out
#SBATCH --error=logs/M_%A_%a.err

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --threads-per-core=1
##SBATCH --ntasks-per-core=1

#SBATCH --mem-per-cpu=2G

#SBATCH --partition=CPU
#SBATCH --gpus-per-task=0
#SBATCH --gpus=0
#SBATCH --time=00-23:59:59

##SBATCH --array=1-1 #1 job to 1 GPU

#export PYTHONPATH=.venv/lib/python3.10/site-packages/:../libs/
#export APPTAINERENV_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
# bnd -exec echo $LD_LIBRARY_PATH

bnd  run --cpu  python  "$@" #> ./out.txt

##gres=gpu:h100