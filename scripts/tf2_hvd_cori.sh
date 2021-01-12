#!/bin/bash
#SBATCH -C knl
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -J train-cori
#SBATCH -o logs/%x-%j.out

module=$1

module load $module
module list

set -x
srun -l -u python horovod/examples/tensorflow2/tensorflow2_synthetic_benchmark.py
