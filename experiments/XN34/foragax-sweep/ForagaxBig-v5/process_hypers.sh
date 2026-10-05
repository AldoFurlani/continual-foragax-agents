#!/bin/bash
#SBATCH --account=aip-amw8
#SBATCH --job-name=XN34_foragax-sweep_ForagaxBig-v5_process_hypers
#SBATCH --mem-per-cpu=128G
#SBATCH --ntasks=1
#SBATCH --output={$SCRATCH}/slurm-%j.out
#SBATCH --time=2:00:00

module load arrow/19

cp -R .venv $SLURM_TMPDIR

export MPLBACKEND=TKAgg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NPROC=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_PLATFORMS=cpu

$SLURM_TMPDIR/.venv/bin/python experiments/XN34/foragax-sweep/ForagaxBig-v5/hypers.py
