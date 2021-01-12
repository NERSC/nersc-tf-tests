#!/bin/bash

set -ex

module=tensorflow/gpu-2.2.0-py37

sbatch -J knl_tf2_hvd_resnet_synthetic -N 1 scripts/tf2_hvd_cori.sh $module
sbatch -J knl_tf2_hvd_resnet_synthetic -N 2 scripts/tf2_hvd_cori.sh $module
