#!/bin/bash

set -ex

# FIXME: temporary workaround for image lookup issue
image=nersc/tensorflow:ngc-21.03-tf2-v0
SID=$(shifterimg lookup $image)

sbatch -N 1 scripts/tf2_hvd_pm_shifter.sh $SID
sbatch -N 2 scripts/tf2_hvd_pm_shifter.sh $SID
