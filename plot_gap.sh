#!/bin/sh
set -eu

dataset="${1:-cora}"
attack_test_size="${2:-0.2}"
max_neighbors="${3:-3}"

python utils/plot_gengap_sampling_strategy.py \
  --csv "results/graph_structure_privacy_results_${dataset}.csv" \
  --dataset "$dataset" \
  --attack_test_size "$attack_test_size" \
  --max_neighbors "$max_neighbors" \
  --output "plots/${dataset}_gengap_sampling_strategy.png"
