#!/usr/bin/env bash
set -euo pipefail
python -m src.features.build_features --data_root data/project_5year --n_workers 4 --run_name baseline_run
python -m src.train.run_experiment --data_root data/project_5year --run_name baseline_run --split_mode simple --models lgbm,ridge,elasticnet,rf,extra_trees,torch_linear,torch_mlp  --use_feat --gpu_id 2
