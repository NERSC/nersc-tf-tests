#!/bin/bash
#SBATCH -J tf-multigpu-test
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH -d singleton
#SBATCH -o logs/%x-%j.out

module list
set -x
which python
nvidia-smi

# DEBUGGING SETTINGS
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3}
export TF_CPP_MIN_VLOG_LEVEL=${TF_CPP_MIN_VLOG_LEVEL:-1}

srun python tests/tf_multigpu.py
