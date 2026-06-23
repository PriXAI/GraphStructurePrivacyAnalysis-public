# Third-Party Dataset Notices

This artifact includes saved train-test splits derived from public benchmark
graph datasets. The artifact's source code is released under the MIT License,
but this does not override any terms that may apply to the underlying
third-party datasets.

## Cora and PubMed

Source: Planetoid preprocessed datasets

Repository: https://github.com/kimiyoung/planetoid

License file for the Planetoid reference implementation:
https://github.com/kimiyoung/planetoid/blob/master/LICENSE

Dataset documentation:
https://github.com/kimiyoung/planetoid#preprocessed-datasets

The Planetoid repository is MIT-licensed and documents included preprocessed
datasets for Cora and PubMed. Users reusing these datasets should also cite:

```bibtex
@inproceedings{yang2016planetoid,
  title     = {Revisiting Semi-Supervised Learning with Graph Embeddings},
  author    = {Yang, Zhilin and Cohen, William W. and Salakhutdinov, Ruslan},
  booktitle = {International Conference on Machine Learning},
  year      = {2016}
}
```

## Chameleon

Source: WikipediaNetwork benchmark data, whose original source is identified
by PyTorch Geometric as:
https://arxiv.org/abs/1909.13021


PyTorch Geometric WikipediaNetwork documentation:
https://pytorch-geometric.readthedocs.io/en/2.5.3/generated/torch_geometric.datasets.WikipediaNetwork.html

 We did not identify a separate explicit dataset
license in the upstream sources. Users should treat Chameleon as third-party
benchmark data and cite the original dataset sources when reusing it.

```bibtex
@ARTICLE{9514682,
  author={Rozemberczki, Benedek and Allen, Carl and Sarkar, Rik and Thilo Gross, xx},
  journal={Journal of Complex Networks}, 
  title={Multi-Scale attributed node embedding}, 
  year={2021},
  volume={9},
  number={1},
  pages={1-22},
  keywords={node embedding;node classification;attributed network;dimensionality reduction},
  doi={10.1093/comnet/cnab014}}
```


