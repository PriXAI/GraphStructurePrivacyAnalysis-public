
Paper title: **Impact of Graph Structure on Membership-Inference Risk for Graph Neural Networks**

Requested Badge(s):
  - [X] **Available**
  - [X] **Functional**
  - [X] **Reproduced**



## Description 
This artifact accompanies the PETS 2026 paper *Impact of Graph Structure on Membership-Inference Risk for Graph Neural Networks* by Megha Khosla. It
contains the code and data needed to reproduce the paper's graph neural network
membership-inference experiments.

The artifact studies how graph structure affects node-level membership privacy
in GNNs. In particular, it evaluates two structural factors: how the training
graph is constructed and what edge information is available at inference time.
For training-graph construction, the artifact compares *snowball sampling*,
a structure-aware procedure, with *uniform random node sampling*. For
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

### Accessibility

The artifact is available through the following persistent GitHub repository:

https://github.com/PriXAI/GraphStructurePrivacyAnalysis-public/tree/main

This repository contains the source code, experiment scripts, environment specification, saved train-test graph splits, and README instructions needed to reproduce the artifact. The artifact should be evaluated from the latest commit on the `main` branch during the review process. After artifact evaluation is finalized, we will provide the artifact chairs with a stable reference, such as a specific commit ID, for archival listing.

### License 

The source code in this artifact is released under the MIT License. Included datasets remain subject to their original licenses.

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


### Hardware Requirements 

No special hardware is required. A standard laptop is sufficient for installing
the environment, running the smoke test, and executing a small one-split
functional check.

For full reproduction of all main experiments, we recommend using a server. The experiments are CPU-compatible and do not require a GPU,
but the full experiment matrix trains multiple target GNNs and
membership-inference attack models across datasets, models, train ratios,
sampling strategies, splits, and attack-test sizes. Running the full
reproduction on a server reduces wall-clock time and avoids interrupting
long-running jobs on a personal laptop.


### Software Requirements 

The artifact is implemented in Python and is intended to run in a Conda
environment. It does not require a VM, Docker container, proprietary software,
or special operating-system packages beyond a standard Python/Conda setup.

The experiments were run with the environment specified in `environment.yml`.
The required software is:

1. Operating system: Linux or macOS with Conda available. The artifact is not tied to a specific operating-system version. We believe that it can be easily run on other OS too but we have not tested that.
2. Environment manager: Conda 
3. Programming language: Python 3.10.
4. All required Python packages are listed in environment.yml file

No pretrained machine-learning model is required. The artifact trains the GNN
target models and membership-inference attack models from scratch during the
experiments. The required graph data and saved train-test splits are included
under `data/`

### Estimated Time and Storage Consumption

The repository checkout requires approximately 1.6 GB of disk space, of which
approximately 1.5 GB is the bundled graph data and saved train-test splits. The
Conda environment requires approximately 1 GB of additional disk space. The CSV
and plot outputs are small, typically below 10 MB. We recommend reserving at
least 5 GB of disk space to allow for the repository, Conda environment, package
caches, PyTorch Geometric cache files, and temporary files.

The expected human effort is approximately 20-30 minutes for cloning the
repository, creating the Conda environment, running the smoke test, and checking
the functional output. Full reproduction requires approximately 30-60 minutes of
human effort to launch the experiment scripts and compare the generated CSV
files and plots with the paper results.

On our local machine, the environment smoke test completed in 8.79 seconds, and
a one-split Cora functional check completed in 42.24 seconds. A full five-split
experiment setting with four attack-test sizes is expected to take
approximately 10-20 minutes, depending on the dataset and model. Reproducing
the full set of main result CSVs may take approximately 8-12 compute-hours on a
standard laptop or CPU workstation. Runtime may vary with CPU, memory, BLAS,
and whether PyTorch Geometric datasets have already been cached.

| Task | Human time | Compute time | Storage |
|---|---:|---:|---:|
| Clone repository | 2-5 min | 1-5 min | 1.6 GB |
| Create Conda environment | 5-10 min | 5-20 min | ~1 GB |
| Environment smoke test | 1-2 min | ~9 sec measured | negligible |
| One-split functional check | 2-5 min | ~42 sec measured | <1 MB |
| One full experiment setting | 2-5 min | ~10-20 min estimated | <1 MB |
| Full main-result reproduction | 30-60 min | ~8-12 hours estimated | <10 MB outputs |

The 8-12 hour of full result reproduction estimate assumes that Cora, PubMed, and Chameleon experiments are run in parallel as separate server jobs.

## Installation

Clone the artifact repository, then create and activate the project
environment:

```bash
git clone https://github.com/PriXAI/GraphStructurePrivacyAnalysis-public.git
cd GraphStructurePrivacyAnalysis-public
conda env create -f environment.yml
conda activate graph-structure-privacy-analysis
```

The `conda env create` command installs Python 3.10 and the Python packages
listed in `environment.yml`, including PyTorch, PyTorch Geometric, NumPy,
Pandas, scikit-learn, NetworkX, Matplotlib, and Seaborn. No further
installation step is needed. The experiment scripts should be run from the root
of the cloned repository so that relative paths such as
`data/cora/snowball_3_0.1/split_1.pt` resolve correctly.

## Testing the Environment

After activating the Conda environment, run the following smoke test from the
repository root:

```bash
python - <<'PY'
import torch
import torch_geometric
import numpy
import pandas
import sklearn
import networkx

from models.model import GNNModel
from utils.data_utils import load_dataset

cora = load_dataset("cora")
split = torch.load("data/cora/snowball_3_0.1/split_1.pt")

assert cora.x.shape[0] > 0
assert split.train_mask.sum().item() > 0
assert split.test_mask.sum().item() > 0

print("Environment OK")
print(f"Cora nodes: {cora.x.shape[0]}")
print(f"Split train nodes: {split.train_mask.sum().item()}")
print(f"Split test nodes: {split.test_mask.sum().item()}")
PY
```

The expected output is:

```text
Environment OK
Cora nodes: 2708
Split train nodes: 270
Split test nodes: 2438
```

Minor differences in package-warning messages are acceptable. If this command
prints `Environment OK`, the core dependencies can be imported and the bundled
graph split can be loaded. The first call to `load_dataset("cora")` may download
the public PyTorch Geometric Planetoid Cora cache if it is not already present.

As a short functional check, run one Cora split:

```bash
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
```

This command trains one target GNN on one saved Cora split, evaluates the model
under the original-graph, full-graph, and no-edge settings, runs the
membership-inference attack for one attack-test-size setting, and writes
`results/environment_check_cora.csv`. Successful completion confirms that the
training, evaluation, attack, and CSV-output pipeline is functioning.

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

Example full five-split run for Cora with train graph constructed using
snowball sampling:

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

For the full experiment matrix, use `runner-general.py`:

```bash
python runner-general.py
```

This runner evaluates Cora, PubMed, and Chameleon for the configured target
models, train ratios, sampling strategies, saved splits, and attack-test sizes.
It writes dataset-specific CSV files under `results/`, for example:

```text
results/graph_structure_privacy_results_cora.csv
results/graph_structure_privacy_results_pubmed.csv
results/graph_structure_privacy_results_chameleon.csv
```

Full reproduction is best run on a server or workstation because it may take
several hours.

## Main Results and Claims

This artifact supports the following main claims from the paper:

1. The training graph construction has an effect on performance gap of the models which in turn affects the membership inference risk. Specifically, snowball sampling strategy to construct training graph often hurts generalization relative to random sampling due to its coverage bias.
2. Inference-time edge access affects both target-model performance and
   membership-inference risk. The artifact evaluates each trained target model
   under the original sampled graph, the full graph, and a no-edge setting.
3. The train-test generalization gap is an incomplete proxy for
   membership-inference risk. The generated CSVs and plots show that membership
   advantage can change independently of the generalization gap.

The CSV columns `gen_gap_orig`, `gen_gap_alledges`, and `gen_gap_noedges`
measure the performance gap when the edges from the original split, all edges of the graph and none of the edges were used respectively during inference. The columns `ma_orig`, `ma_transductive`, and
`ma_nograph` measure membership advantage for the corresponding edge-access
settings.

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

```bash
python utils/plot_3panel_attack_test_sizes.py \
  --sampling snowball \
  --csv results/graph_structure_privacy_results_cora.csv \
        results/graph_structure_privacy_results_chameleon.csv \
        results/graph_structure_privacy_results_pubmed.csv \
  --output plots/three_panel_perf_vs_adv_attack_test_sizes_snowball.png
```

## Limitations

The artifact trains target models and attack models from scratch rather than
shipping pretrained checkpoints. This keeps the artifact self-contained, but it
means full reproduction takes several hours and is better run on a server or
workstation.

Small numerical
differences may occur across machines and software backends even with fixed
random seeds. The expected qualitative trends should remain stable.

The artifact includes the saved train-test splits used by the experiments. It
does not attempt to reproduce every possible random split or every possible
hyperparameter setting beyond the experiment matrix described above.

## Notes on Reusability

The code can be reused to evaluate additional graph datasets, GNN architectures,
sampling strategies, and membership-inference attacks. The main extension
points are:

- `models/model.py` for target GNN architectures.
- `attacks/` for membership-inference attacks.
- `utils/data_utils.py` for dataset loading and split generation.
- `runner-general.py` for configuring experiment batches.

New saved splits can be generated with `create_and_save_splits_joint` and then
passed to `main.py` using the `--input_path` argument.
