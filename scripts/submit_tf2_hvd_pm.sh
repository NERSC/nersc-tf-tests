#!/bin/bash

acct=$1

# Setup software
module load tensorflow/2.9.0

# Single node, 4 gpus
sbatch -N 1 -A $acct scripts/tf2_hvd_pm.sh

# Two nodes, 8 gpus
sbatch -N 2 -A $acct scripts/tf2_hvd_pm.sh
