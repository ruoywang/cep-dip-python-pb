#!/bin/bash
#SBATCH -J pb_py_timing5
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o /scratch/08384/tg876840/tmp/b-1D_PB/pure_python/results/from_chgcar_newton_detail5/job.o
#SBATCH -e /scratch/08384/tg876840/tmp/b-1D_PB/pure_python/results/from_chgcar_newton_detail5/job.e
#SBATCH -D /scratch/08384/tg876840/tmp/b-1D_PB

export OMP_NUM_THREADS=1
export PB_PROFILE_INNER=1

python -m pure_python.solve_from_chgcar_newton \
  --config pure_python/configs/cal18.json \
  --chgcar data/case_cal18/CHGCAR \
  --potcar data/case_cal18/POTCAR \
  --phi-ref data/case_cal18/PHI \
  --rhob-ref data/case_cal18/RHOB \
  --rhoion-ref data/case_cal18/RHOION \
  --out-dir pure_python/results/from_chgcar_newton_detail5 \
  --fixsol-steps 5 \
  --tol 1.0e-3 \
  --max-outer 20 \
  --cg-max-iter 200
