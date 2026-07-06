#!/bin/sh
set -eu

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
