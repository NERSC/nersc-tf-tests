#!/bin/bash

set -ex

image=nersc/tensorflow:ngc-20.08-tf2-v1

sbatch --image $image -J tf2_hvd_resnet_synthetic -N 1 -n 2 scripts/tf2_hvd_cgpu_shifter.sh
sbatch --image $image -J tf2_hvd_resnet_synthetic -N 2 scripts/tf2_hvd_cgpu_shifter.sh
