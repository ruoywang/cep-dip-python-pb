#!/bin/bash
#SBATCH -J pb_fastlf_f
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH -t 02:00:00
#SBATCH -o pure_python/results/from_chgcar_newton_fastlf_full/job.o
#SBATCH -e pure_python/results/from_chgcar_newton_fastlf_full/job.e
#SBATCH -D /scratch/08384/tg876840/tmp/b-1D_PB

export PB_FFT_BACKEND=scipy
export PB_FFT_WORKERS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p pure_python/results/from_chgcar_newton_fastlf_full

python -m pure_python.solve_from_chgcar_newton \
  --config pure_python/configs/cal18.json \
  --chgcar data/case_cal18/CHGCAR \
  --potcar data/case_cal18/POTCAR \
  --phi-ref data/case_cal18/PHI \
  --rhob-ref data/case_cal18/RHOB \
  --rhoion-ref data/case_cal18/RHOION \
  --out-dir pure_python/results/from_chgcar_newton_fastlf_full \
  --fixsol-steps 5 \
  --tol 1.0e-3 \
  --max-outer 12 \
  --cg-max-iter 40
