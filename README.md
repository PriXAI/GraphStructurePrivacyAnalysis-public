# Artifact Appendix (Required for all badges)

Paper title: **Impact of Graph Structure on Membership-Inference Risk for Graph Neural Networks**

Requested Badge(s):
  - [X] **Available**
  - [X] **Functional**
  - [X] **Reproduced**

## Description (Required for all badges)
This artifact accompanies the PETS 2026 paper Impact of Graph Structure on Membership-Inference Risk for Graph Neural Networks by Megha Khosla. It contains the code and data needed to reproduce the paper's graph neural network membership-inference experiments.

The artifact studies how graph structure affects node-level membership privacy in GNNs. In particular, it evaluates two structural factors: how the training graph is constructed and what edge information is available at inference time. For training-graph construction, the artifact compares snowball sampling, a structure-aware procedure, with uniform random node sampling. For inference-time edge access, it evaluates the same target model under the original sampled graph, the full graph, and a no-edge setting.

The released scripts train target GNNs, run membership-inference attacks, and generate the main result plots. The included data consists of the saved train-test graph splits used by the experiments, including the postprocessed Chameleon graph with self-loops removed and edges made undirected. Together, the code and data support the paper's central finding: graph structure directly shapes membership-inference risk, and the train-test generalization gap is an incomplete proxy for that risk because membership advantage can change independently of the generalization gap.

### Security/Privacy Issues and Ethical Concerns (Required for all badges)

This artifact does not introduce any known security risk to the evaluator's machine. It does not disable any operating-system, network, or software security mechanism, and it does not run any unsafe code. The artifact consists of scripts for training graph neural networks, running membership-inference evaluations, and generating plots.

The experiments use public benchmark graph data and saved train-test splits. They do not include user-study data, private personal data, anonymized transcripts, survey responses, or other human-subjects data. Therefore, no additional ethical-review was required for releasing the artifact. The privacy analysis is limited to measuring membership-inference risk in this public-data experimental setting.

## Basic Requirements (Required for Functional and Reproduced badges)

### Hardware Requirements

No special hardware is required. A standard laptop is sufficient for installing
the environment, running the smoke test, and executing a small one-split
functional check.

For full reproduction of all main experiments, we recommend using a server. The
experiments are CPU-compatible and do not require a GPU, but the full experiment
matrix trains multiple target GNNs and membership-inference attack models
across datasets, models, train ratios, sampling strategies, splits, and
attack-test sizes. Running the full reproduction on a server reduces wall-clock
time and avoids interrupting long-running jobs on a personal laptop.

The experiments reported in the paper were run as CPU jobs on the Delft AI
Cluster (DAIC), a Slurm-managed computing cluster at TU Delft. The DAIC node on
which the recorded jobs ran had 1 x AMD EPYC 7502P 32-Core Processor, 32
physical cores, a 2.5 GHz clock, 251 GB of RAM, and 148 GB of local temporary
storage. For PubMed jobs we requested 4 GB of memory; all other jobs we requested 1 GB.

### Software Requirements (Required for Functional and Reproduced badges)

The artifact is implemented in Python and is intended to run in a Conda
environment. It does not require a VM, Docker container, proprietary software,
or special operating-system packages beyond a standard Python/Conda setup. The
artifact was tested on Linux and macOS with Conda. The full experiments were run on x86_64 Linux CPU nodes on the Delft AI Cluster.

The experiments were run with the environment specified in `environment.yml`.
The required software is:

1. Operating system: Linux or macOS with Conda available. The artifact is not
   tied to a specific operating-system version. It may also run on other
   operating systems, but we have not tested them.
2. Environment manager: Conda.
3. Programming language: Python 3.10.
4. All required Python packages are listed in the `environment.yml` file.

No pretrained machine-learning model is required. The artifact trains the GNN
target models and membership-inference attack models from scratch during the
experiments. The required graph data and saved train-test splits are included
under `data/`

#### Data
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

#### Models

Available target GNN model types:

- `GCN`
- `GraphSage`
- `GAT`

The model implementation is in `models/model.py`. The main experiment orchestration is in `main.py`.


### Estimated Time and Storage Consumption (Required for Functional and Reproduced badges)

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
approximately 10-20 minutes for smaller datasets such as Cora and faster models
such as GCN. Larger datasets such as PubMed and slower models such as GAT may
take longer. Reproducing the full set of main result CSVs may take
approximately 8-12 compute-hours on a standard laptop or CPU workstation when
the three datasets are run in parallel; a fully sequential run can take longer.
Runtime may vary with CPU, memory, BLAS, and whether PyTorch Geometric datasets
have already been cached. Once the CSV files have been generated, the plotting
commands are lightweight: each plot typically takes seconds to a few minutes,
and all main plots should take less than 10 human-minutes to launch and inspect.


## Environment (Required for all badges)

The artifact is distributed as a Git repository. All source code, the Conda
environment specification, and the saved graph train-test split files required
by the experiments are included in the repository. No separate download,
pretrained model, VM image, Docker image, or proprietary software is required.

### Accessibility (Required for all badges)

The artifact is available through the following persistent GitHub repository:

https://github.com/PriXAI/GraphStructurePrivacyAnalysis-public/tree/main

This repository contains the source code, experiment scripts, environment specification, saved train-test graph splits, and README instructions needed to reproduce the artifact. The artifact should be evaluated from the latest commit on the `main` branch during the review process. After artifact evaluation is finalized, we will provide the artifact chairs with a stable reference, such as a specific commit ID, for archival listing.

The repository contains:

- `environment.yml`: the Conda environment specification.
- `main.py`: the main experiment runner.
- `runner-*.py`: convenience scripts for selected experiment batches.
- `attacks/`, `models/`, `training/`, and `utils/`: implementation modules.
- `data/`: saved graph splits for Cora, PubMed, and Chameleon, plus the
  processed Chameleon graph used by the experiments.

### License
The source code in this artifact is released under the MIT License. See
[LICENSE](LICENSE).

Included datasets and saved graph splits remain subject to their original
dataset terms. See [THIRD_PARTY_DATASETS.md](THIRD_PARTY_DATASETS.md) for
dataset provenance, upstream license/terms notes, and citation guidance.

### Set up the environment (Required for Functional and Reproduced badges)

Install Conda if it is not already available. Then clone the
artifact repository and create the project environment:

```bash
git clone https://github.com/PriXAI/GraphStructurePrivacyAnalysis-public.git
cd GraphStructurePrivacyAnalysis-public
conda env create -f environment.yml
conda activate graph-structure-privacy-analysis
```

The `conda env create` command installs Python 3.10 and the Python packages
listed in `environment.yml`, including PyTorch, PyTorch Geometric, NumPy,
Pandas, scikit-learn, NetworkX, Matplotlib, and Seaborn.

No further installation step is needed. The experiment scripts should be run
from the root of the cloned repository so that relative paths such as
`data/cora/snowball_3_0.1/split_1.pt` resolve correctly.

### Testing the Environment (Required for Functional and Reproduced badges)

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
graph data can be loaded. 

As an optional functional check, run a short one-split experiment:

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

This command trains one target GNN on one saved Cora split, runs the
membership-inference attack for one attack-test-size setting, and writes
`results/environment_check_cora.csv`. Successful completion confirms that the
training, evaluation, attack, and CSV-output pipeline is functioning.

### Docker Environment (Optional)

The primary supported setup is the Conda environment above. The artifact also
includes a `Dockerfile` for reviewers who prefer to build an isolated Docker
environment. Build the image from the repository root:

```bash
docker build -t graph-structure-privacy-analysis .
```

To smoke-test the Docker image without running a full experiment, run:

```bash
docker run --rm graph-structure-privacy-analysis \
  conda run --no-capture-output -n graph-structure-privacy-analysis \
  python -c "import torch, torch_geometric, pandas, sklearn, networkx; split=torch.load('data/cora/snowball_3_0.1/split_1.pt'); print('Docker smoke test OK'); print('train nodes:', split.train_mask.sum().item()); print('test nodes:', split.test_mask.sum().item())"
```

To run the one-split functional check inside Docker and write the output CSV to
the host `results/` directory, run:

```bash
docker run --rm \
  -v "$PWD/results:/artifact/results" \
  graph-structure-privacy-analysis \
  conda run --no-capture-output -n graph-structure-privacy-analysis \
  python main.py \
    --dataset_name cora \
    --model_type GCN \
    --train_ratio 0.1 \
    --num_splits 1 \
    --max_neighbors 3 \
    --random_seed 42 \
    --strategy snowball \
    --input_path data/cora/snowball_3_0.1 \
    --output_path results/docker_environment_check_cora.csv \
    --attack_test_size 0.2
```

The Docker image was smoke-tested by importing the core dependencies and
loading the bundled Cora split. On an Apple Silicon machine, Docker built a
`linux/aarch64` image; on a typical x86_64 server, Docker will build a
`linux/amd64` image.



## Artifact Evaluation (Required for Functional and Reproduced badges)

### Main Results and Claims

#### Main Result 1: Effect of Graph Sampling and Edge Access on Performance Gap

We compare the effect of two training-graph sampling strategies, snowball
sampling and uniform random node sampling, on the observed train-test
performance gap of different GNN models.

#### Main Result 2: Performance Gap and Membership Advantage

We compare train-test generalization gap with membership advantage across graph
sampling strategies and inference-time edge-access settings. The artifact
supports the claim that membership advantage can vary independently of the
generalization gap.


### Experiments

#### Main Experiment to Reproduce All Results

To reproduce the claims supported by the artifact, run the one-dataset
experiment runner separately for each dataset. We recommend launching the three
dataset runs as parallel server jobs:

```bash
python runner-one-dataset.py cora
python runner-one-dataset.py pubmed
python runner-one-dataset.py chameleon
```

Each command runs the configured experiment matrix for one dataset, including
the target models, train ratios, sampling strategies, saved splits, and
attack-test sizes. The commands generate the following main result CSV files
under `results/`:

```text
results/graph_structure_privacy_results_cora.csv
results/graph_structure_privacy_results_pubmed.csv
results/graph_structure_privacy_results_chameleon.csv
```

##### Result CSVs

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


Docker commands for the main experiment are given below. First build the image
as described in the Docker setup section. Then, to reproduce all results with
one dataset per container, launch the following commands. The `results/`
directory is mounted so that the generated CSV files are written back to the
host:

```bash
docker run --rm -v "$PWD/results:/artifact/results" graph-structure-privacy-analysis \
  conda run --no-capture-output -n graph-structure-privacy-analysis \
  python runner-one-dataset.py cora

docker run --rm -v "$PWD/results:/artifact/results" graph-structure-privacy-analysis \
  conda run --no-capture-output -n graph-structure-privacy-analysis \
  python runner-one-dataset.py pubmed

docker run --rm -v "$PWD/results:/artifact/results" graph-structure-privacy-analysis \
  conda run --no-capture-output -n graph-structure-privacy-analysis \
  python runner-one-dataset.py chameleon
```

These three Docker commands can be run in parallel on a suitable server. To run
all experiments sequentially inside one Docker container instead, use:

```bash
docker run --rm \
  -v "$PWD/results:/artifact/results" \
  graph-structure-privacy-analysis \
  conda run --no-capture-output -n graph-structure-privacy-analysis \
  python runner-general.py
```



#### Sub-Experiment/ Result 1: Effect of train graph construction strategy on Performance Gap

Use `utils/plot_gengap_sampling_strategy.py` to compare random and snowball
sampling for one dataset while holding `Use_Loss`, attack test size, and the
max-neighbor setting fixed.

```bash
python utils/plot_gengap_sampling_strategy.py \
  --csv results/graph_structure_privacy_results_cora.csv \
  --dataset cora \
  --attack_test_size 0.2 \
  --max_neighbors 3 \
  --output plots/cora_gengap_sampling_strategy.png
```

This command reads the Cora result CSV and generates a plot comparing the
train-test generalization gap for snowball sampling and random sampling. The
`attack_test_size` argument is used only to select rows from the result CSV; it
does not affect the target-model performance gap. Therefore, any available
attack-test size in the CSV, such as `0.9`, `0.8`, `0.5`, or `0.2`, can be used
for this plot. This plot corresponds to Figure 4 in the paper. To reproduce
Figures 5 and 6, replace the dataset name with `pubmed` and `chameleon`,
respectively. After the result CSVs are available, each of these plotting
commands is expected to take only seconds to a few minutes.

#### Sub-Experiment/ Result 2: Effect of Performance Gap on Membership Advantage

Use `utils/plot_3panel_attack_test_sizes.py` to plot train-test performance gap
against membership advantage across datasets, graph-access settings, models,
train ratios, and attack-test sizes. This sub-experiment supports the claim
that membership advantage does not always monotonically increase with the
train-test performance gap.

Snowball sampling:

```bash
python utils/plot_3panel_attack_test_sizes.py \
  --sampling snowball \
  --csv results/graph_structure_privacy_results_cora.csv \
        results/graph_structure_privacy_results_chameleon.csv \
        results/graph_structure_privacy_results_pubmed.csv \
  --output plots/three_panel_perf_vs_adv_attack_test_sizes_snowball.png
```

Random sampling:

```bash
python utils/plot_3panel_attack_test_sizes.py \
  --sampling random \
  --csv results/graph_structure_privacy_results_cora.csv \
        results/graph_structure_privacy_results_chameleon.csv \
        results/graph_structure_privacy_results_pubmed.csv \
  --output plots/three_panel_perf_vs_adv_attack_test_sizes_random.png
```

Each command reads the generated result CSV files and creates a multi-panel
plot. The x-axis shows the train-test performance gap, and the y-axis shows
membership advantage. Points are grouped by dataset, attack-test size, and
inference-time graph-access setting: original sampled graph, full graph, and
no-edge access. The two plots correspond to Figures 7 and 8 in the paper. After the result CSVs are available, each plotting command is
expected to take only seconds to a few minutes.




## Limitations

The artifact trains target models and attack models from scratch rather than
shipping pretrained checkpoints. This keeps the artifact self-contained, but it
means full reproduction takes several hours and is better run on a server or
workstation.

Small numerical differences may occur across machines and software backends
even with fixed random seeds. The expected qualitative trends should remain
stable.

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
