#!/bin/bash
#SBATCH -J pb_rhob_align
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o pure_python/results/ref_phi_diag/rhob_alignment.o
#SBATCH -e pure_python/results/ref_phi_diag/rhob_alignment.e
#SBATCH -D /scratch/08384/tg876840/tmp/b-1D_PB

python -m pure_python.diagnose_rhob_alignment \
  --config pure_python/configs/cal18.json \
  --out pure_python/results/ref_phi_diag/rhob_alignment.txt
