#!/bin/bash

set -ex

image=nersc/tensorflow:ngc-21.03-tf2-v0

sbatch -N 1 --image=$image scripts/tf2_hvd_pm_shifter.sh
sbatch -N 2 --image=$image scripts/tf2_hvd_pm_shifter.sh
