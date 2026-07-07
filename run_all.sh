#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1,2
python run_experiment.py -c configs/faust_shape_matching.yaml
python run_experiment.py -c configs/scape_shape_matching.yaml
python run_experiment.py -c configs/smal_shape_matching.yaml
