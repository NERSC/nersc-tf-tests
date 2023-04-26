#!/bin/bash

acct=$1
set -ex

image=nersc/tensorflow:ngc-23.03-tf2-v0

sbatch -N 1 -A $acct --image=$image scripts/tf2_hvd_pm_shifter.sh
sbatch -N 2 -A $acct --image=$image scripts/tf2_hvd_pm_shifter.sh
