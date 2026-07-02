#!/bin/bash
#SBATCH -J pb_sign_minus
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:20:00
#SBATCH -o pure_python/results/from_chgcar_sign_minus/job.o
#SBATCH -e pure_python/results/from_chgcar_sign_minus/job.e
#SBATCH -D /scratch/08384/tg876840/tmp/b-1D_PB

python -m pure_python.solve_from_chgcar_simple \
  --config pure_python/configs/cal18.json \
  --out-dir pure_python/results/from_chgcar_sign_minus \
  --iterations 3 \
  --mix 0.3 \
  --source-sign -1.0
