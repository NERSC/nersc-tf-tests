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

#SBATCH --volume="/dev/infiniband:/sys/class/infiniband_verbs"

set -x
srun -l -u shifter python horovod/examples/tensorflow2/tensorflow2_synthetic_benchmark.py
