#!/bin/bash
#SBATCH -J pb_cdip_m
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o pure_python/results/from_chgcar_cdipol_minus/job.o
#SBATCH -e pure_python/results/from_chgcar_cdipol_minus/job.e
#SBATCH -D /scratch/08384/tg876840/tmp/b-1D_PB

python -m pure_python.solve_from_chgcar_cdipol \
  --config pure_python/configs/cal18.json \
  --chgcar data/case_cal18/CHGCAR \
  --potcar data/case_cal18/POTCAR \
  --phi-ref data/case_cal18/PHI \
  --rhob-ref data/case_cal18/RHOB \
  --rhoion-ref data/case_cal18/RHOION \
  --out-dir pure_python/results/from_chgcar_cdipol_minus \
  --iterations 4 \
  --mix 0.5 \
  --source-sign -1
