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

# FIXME: temporary workaround for image lookup, must pass in image ID
SID=$1

export NCCL_DEBUG=INFO

# Funny workaround to prevent shifter directory confusion.
# UPDATE THIS AS NEEDED FOR YOUR WORKING PATH.
cd /home/$USER/tensorflow-build/nersc-tf-tests

srun -l -u shifter --image=id:${SID} --module gpu \
    python horovod/examples/tensorflow2/tensorflow2_synthetic_benchmark.py
