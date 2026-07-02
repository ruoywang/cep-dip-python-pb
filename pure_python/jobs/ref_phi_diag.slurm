#!/bin/bash
#SBATCH -J pb_refphi_diag
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o pure_python/results/ref_phi_diag/job.o
#SBATCH -e pure_python/results/ref_phi_diag/job.e
#SBATCH -D /scratch/08384/tg876840/tmp/b-1D_PB

python -m pure_python.diagnose_with_reference_phi \
  --config pure_python/configs/cal18.json \
  --out-dir pure_python/results/ref_phi_diag
