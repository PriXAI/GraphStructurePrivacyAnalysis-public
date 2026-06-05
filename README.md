
Paper title: **Impact of Graph Structure on Membership-Inference Risk for Graph Neural Networks**

Requested Badge(s):
  - [X] **Available**



## Description 
This artifact accompanies the PETS 2026 paper \emph{Impact of Graph Structure
on Membership-Inference Risk for Graph Neural Networks} by Megha Khosla. It
contains the code and data needed to reproduce the paper's graph neural network
membership-inference experiments.

The artifact studies how graph structure affects node-level membership privacy
in GNNs. In particular, it evaluates two structural factors: how the training
graph is constructed and what edge information is available at inference time.
For training-graph construction, the artifact compares \emph{snowball sampling},
a structure-aware procedure, with \emph{uniform random node sampling}. For
inference-time edge access, it evaluates the same target model under the
original sampled graph, the full graph, and a no-edge setting.

The released scripts train target GNNs, run membership-inference attacks, and
generate the main result plots. The included data consists of the saved
train-test graph splits used by the experiments, including the postprocessed
Chameleon graph with self-loops removed and edges made undirected. Together,
the code and data support the paper's central finding: graph structure directly
shapes membership-inference risk, and the train-test generalization gap is an
incomplete proxy for that risk because membership advantage can change
independently of the generalization gap.

### Security/Privacy Issues and Ethical Concerns 

This artifact does not introduce any known security risk to the evaluator's
machine. It does not disable any operating-system, network, or software security
mechanism, and it does not run any unsafe code. The artifact consists of scripts for training graph neural networks, running membership-inference evaluations, and
generating plots.

The experiments use public benchmark graph data and saved train-test splits.
They do not include user-study data, private personal data, anonymized
transcripts, survey responses, or other human-subjects data. Therefore, no
additional ethical-review or IRB process was required for releasing the
artifact. The privacy analysis is limited to measuring membership-inference
risk in this public-data experimental setting.

## Basic Requirements 

For both sections below, if you are giving reviewers remote access to special
hardware (e.g., Intel SGX v2.0) or proprietary software (e.g., Matlab R2025a)
for the purpose of the artifact evaluation, do not provide these instructions
here but rather in the corresponding submission field on HotCRP.

### Hardware Requirements 

Can run on a laptop (No special hardware requirements). 


### Software Requirements 

The artifact is implemented in Python and is intended to run in a Conda
environment. It does not require a VM, Docker container, proprietary software,
or special operating-system packages beyond a standard Python/Conda setup.

The experiments were run with the environment specified in `environment.yml`.
The required software is:

1.  The artifact is not tied to a specific operating-system  as long as the conda environment can be set-up.
2. Environment manager: Conda 
3. Programming language: Python 3.10.
4. All required Python packages are listed in environment.yml file

No pretrained machine-learning model is required. The artifact trains the GNN
target models and membership-inference attack models from scratch during the
experiments. The required graph data and saved train-test splits are included
under `data/`

## Installation

Create and activate the project environment:

```bash
conda env create -f environment.yml
conda activate graph-structure-privacy-analysis
```

## Data
The inductive train-test splits are created using the two sampling strategies and saved as PyTorch Geometric split objects under `data/`. The experiment scripts expect one split file per split, for example:

```text
data/cora/snowball_3_0.1/split_1.pt
data/cora/random_0.5/split_1.pt
```

Note that the Chameleon base graph is postprocessed to remove self-loops and make the graph undirected. The processed graph is stored at `data/chameleon`, and the corresponding split files are stored under `data/chameleon1/`.

Snowball split directories include the max-neighbor setting in the folder name:

```text
snowball_<max_neighbors>_<train_ratio>
```

Random split directories use:

```text
random_<train_ratio>
```

To generate new splits, use `create_and_save_splits_joint` from `utils/data_utils.py`:

```python
from utils.data_utils import create_and_save_splits_joint, load_dataset

cora = load_dataset("cora")
create_and_save_splits_joint(
    cora,
    num_splits=5,
    max_neighbors=3,
    train_ratio=0.1,
    save_dir="data/cora/snowball_3_0.1",
    strategy="snowball",
)
```

## Models

Available target GNN model types:

- `GCN`
- `GraphSage`
- `GAT`

The model implementation is in `models/model.py`. The main experiment orchestration is in `main.py`.

## Running Experiments

Example run for Cora with train graph constructed using snowball sampling:

```bash
python main.py \
  --dataset_name cora \
  --model_type GCN \
  --train_ratio 0.1 \
  --num_splits 5 \
  --max_neighbors 3 \
  --random_seed 42 \
  --strategy snowball \
  --input_path data/cora/snowball_3_0.1 \
  --output_path results/graph_structure_privacy_results_cora.csv \
  --attack_test_size 0.9 0.8 0.5 0.2
```



The script trains one target GNN per split, evaluates it under the three graph-access settings, and reruns the attack model for each requested `attack_test_size`.

## Result CSVs

Main experiment outputs are CSV files under the 'output_path' provided while running the main experiment. 

Each row in the results file corresponds to one dataset/model/train-ratio/sampling-strategy/attack-test-size setting. Values aggregated across splits are stored as `mean \pm std`.

Important columns:

- `Dataset Name`, `Model_type`, `Train Ratio`, `Strategy`, `Attack Test Size`: experiment setting.
- `Use_Loss`: whether the attack model input used each node's posterior vector appended with its loss value as in the paper, at the moment it is always set to True.
- `train_acc_orig`, `test_acc_orig`: target accuracy on the original sampled split.
- `train_acc_alledges`, `test_acc_alledges`: target accuracy with the full graph.
- `train_acc_noedges`, `test_acc_noedges`: target accuracy with no edges.
- `gen_gap_orig`, `gen_gap_alledges`, `gen_gap_noedges`: train-test performance gap in percent.
- `ma_orig`, `ma_transductive`, `ma_nograph`: membership advantage for original-split, full-graph, and no-edge access.
- `attack_acc_*`, `attack_auc_*`, `attack_ap_*`: attack accuracy, AUC, and average precision.


## Plotting Results

Plots are written under `plots/`.

### Generalization Gap by Sampling Strategy

Use `utils/plot_gengap_sampling_strategy.py` to compare random and snowball sampling for one dataset while holding `Use_Loss`, attack test size, and max-neighbor setting fixed.

```bash
python utils/plot_gengap_sampling_strategy.py \
  --csv results/graph_structure_privacy_results_cora.csv \
  --dataset cora \
  --attack_test_size 0.2 \
  --max_neighbors 3 \
  --output plots/cora_gengap_sampling_strategy.png
```



### Performance Gap vs Membership Advantage

Use `utils/plot_3panel_attack_test_sizes.py` to plot performance gap against membership advantage across attack test sizes.

Snowball sampling:

```bash
python utils/plot_3panel_attack_test_sizes.py \
  --sampling snowball \
  --output plots/three_panel_perf_vs_adv_attack_test_sizes_snowball.png
```

Random sampling:

```bash
python utils/plot_3panel_attack_test_sizes.py \
  --sampling random \
  --output plots/three_panel_perf_vs_adv_attack_test_sizes_random.pdf
```

By default, the script reads:

```text
results/graph_structure_privacy_results_cora.csv
results/graph_structure_privacy_results_chameleon.csv
results/graph_structure_privacy_results_pubmed.csv
```

You can pass explicit CSVs with `--csv`:
