#!/bin/bash

# Setup software
# FIXME: replace with Perlmutter module path when available
source ../setup.sh
conda activate tf-2.4.1

# Single node, 4 gpus
sbatch -N 1 scripts/tf2_hvd_pm.sh

# Two nodes, 8 gpus
sbatch -N 2 scripts/tf2_hvd_pm.sh
