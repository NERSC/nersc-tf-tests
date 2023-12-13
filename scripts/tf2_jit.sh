#!/bin/bash
#SBATCH -J tf-jit-test
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH -o logs/%x-%j.out

module list
set -x
which python
nvidia-smi

# DEBUGGING SETTINGS
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-0} #3
export TF_CPP_MIN_VLOG_LEVEL=${TF_CPP_MIN_VLOG_LEVEL:-0} #1
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"

srun python tests/tf_jit.py
