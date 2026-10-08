#!/bin/bash
#SBATCH -c 16
#SBATCH --mem=150G
#SBATCH -t 24:00:00
#SBATCH -o %x.%A_%a.log
# Generic SLURM wrapper: run one command of the pipeline inside a conda environment.
#
#   sbatch [-p PARTITION] [-J name] [--gres=gpu:1] [--array=0-9] pipeline/slurm/run.sbatch ENV CMD...
#
# ENV is a conda environment name (looked up under $CONDA_ENVS_DIR, default ~/.conda/envs) or the full prefix of an
# environment. Examples (from pipeline/):
#   sbatch slurm/run.sbatch prime python prime_analysis/parameter_grid.py immune sensitivity
#   sbatch --array=0-9 slurm/run.sbatch prime python prime_analysis/parameter_grid.py dlpfc sensitivity --ntasks 10
#   sbatch --gres=gpu:1 slurm/run.sbatch staligner python spatial/run_methods.py staligner default
# With --array, parameter_grid.py reads its task index from SLURM_ARRAY_TASK_ID; pass --ntasks to match the array size.
set -euo pipefail
ENV=$1; shift
[[ $ENV == */* ]] && PREFIX=$ENV || PREFIX=${CONDA_ENVS_DIR:-$HOME/.conda/envs}/$ENV
export PATH="$PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}"      # the environment's own libstdc++ first
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "host=$(hostname) job=$SLURM_JOB_ID task=${SLURM_ARRAY_TASK_ID:-NA} cpus=$SLURM_CPUS_PER_TASK start=$(date -Is)"
"$@"
echo "end=$(date -Is)"
