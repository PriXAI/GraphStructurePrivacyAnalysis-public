#!/bin/sh
set -eu

python main.py \
  --dataset_name cora \
  --model_type GCN \
  --train_ratio 0.1 \
  --num_splits 1 \
  --max_neighbors 3 \
  --random_seed 42 \
  --strategy snowball \
  --input_path data/cora/snowball_3_0.1 \
  --output_path results/environment_check_cora.csv \
  --attack_test_size 0.2
