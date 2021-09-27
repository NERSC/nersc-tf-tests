#!/bin/bash
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-task 1
#SBATCH -t 0:05:00
#SBATCH -J tf2-benchmark-pm-shifter
#SBATCH -d singleton
#SBATCH -o logs/%x-%j.out

set -x

export NCCL_DEBUG=INFO

srun -l -u shifter \
    bash -c "python horovod/examples/tensorflow2/tensorflow2_synthetic_benchmark.py"
