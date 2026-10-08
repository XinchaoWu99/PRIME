#!/bin/bash
# Execute the figure notebooks, one SLURM job (CPU only) per notebook.
#   bash notebooks/run_all.sh [notebook ...]        # from anywhere; default: all five
# Environment: NB_PYTHON  a Python with nbformat + nbclient (default: python)
#              NB_KERNEL  the Jupyter kernel of the analysis environment (default: python3)
#              SBATCH_OPTS  extra sbatch options, e.g. "-p <partition>"
# The memory / time requests cover a first run; later runs reuse the cached UMAP coordinates and take minutes.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${NB_PYTHON:-python}
KERNEL=${NB_KERNEL:-python3}
mkdir -p figures/_logs
declare -A RES=( [01_human_immune]="--mem=96G -t 6:00:00" [02_NSCLC_lung_adenocarcinoma]="--mem=80G -t 8:00:00"
                 [03_DLPFC]="--mem=48G -t 3:00:00" [04_Jurkat_293T]="--mem=16G -t 1:00:00"
                 [05_VisiumHD_mouse_brain]="--mem=48G -t 2:00:00" )
NBS=("$@"); [ ${#NBS[@]} -eq 0 ] && NBS=(01_human_immune 02_NSCLC_lung_adenocarcinoma 03_DLPFC 04_Jurkat_293T 05_VisiumHD_mouse_brain)
for nb in "${NBS[@]}"; do
  nb=${nb%.ipynb}; nb=${nb#notebooks/}
  sbatch --parsable -c 16 ${RES[$nb]} ${SBATCH_OPTS:-} -J "nb_$nb" -o "figures/_logs/%x.%j.log" \
    --wrap "export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16; $PY notebooks/execute_notebook.py notebooks/$nb.ipynb $KERNEL"
done
