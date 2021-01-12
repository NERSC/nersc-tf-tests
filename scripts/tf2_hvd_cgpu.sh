#!/bin/bash
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-task 10
#SBATCH --gpus-per-task 1
#SBATCH -t 0:05:00
#SBATCH -J tf-benchmark
#SBATCH -d singleton
#SBATCH -o logs/%x-%j.out
#SBATCH -A m1759

module=$1

module load $module
module list

set -x
srun -l -u python horovod/examples/tensorflow2/tensorflow2_synthetic_benchmark.py
