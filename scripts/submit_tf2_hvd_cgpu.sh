#!/bin/bash

set -ex

module=tensorflow/gpu-2.2.0-py37

sbatch -J tf2_hvd_resnet50_synthetic -N 1 -n 2 scripts/tf2_hvd_cgpu.sh $module
sbatch -J tf2_hvd_resnet50_synthetic -N 2 scripts/tf2_hvd_cgpu.sh $module
