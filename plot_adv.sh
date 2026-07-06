#!/bin/sh
set -eu

sampling="${1:-snowball}"

python utils/plot_3panel_attack_test_sizes.py \
  --sampling "$sampling" \
  --csv results/graph_structure_privacy_results_cora.csv \
        results/graph_structure_privacy_results_chameleon.csv \
        results/graph_structure_privacy_results_pubmed.csv \
  --output "plots/three_panel_perf_vs_adv_attack_test_sizes_${sampling}.png"
