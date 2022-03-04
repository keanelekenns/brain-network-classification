# contrast-subgraph
This repository contains a replication of "Explainable Classification of Brain Networks via Contrast Subgraphs" by Lanciano et al. (referred to as "the original authors" from here on). The repository for their paper (referred to as "the paper" from here on) can be found [here](https://github.com/tlancian/contrast-subgraph).

Significant modifications have been made to the original authors' code. For example, this repository contains a program named cs_classification.py for evaluating the classifier described in the paper.

Additionally, a new method was attempted in extreme_edge_classification.py, but was found to be ineffective.

## Usage

### cs_classification.py
```
usage: cs_classification.py [-h] [-p {1,2}] [-k K] [-a A] A_dir B_dir alpha

Graph Classification via Contrast Subgraphs

positional arguments:
  A_dir       Filepath to class A directory containing brain network files
  B_dir       Filepath to class B directory containing brain network files
  alpha       Penalty value for contrast subgraph size (varies from 0 to 1)

optional arguments:
  -h, --help  show this help message and exit
  -p {1,2}    Problem Forumlation (default: 1)
  -k K        Number of times to fold data in k-fold cross validation (default: 5)
  -a A        A secondary alpha value to use for the contrast subgraph from B to A (only applies if problem formulation is 1). Note that the original alpha is used for both contrast subgraphs if this is not provided.
```

The output of this program is a report on the average accuracy, precision, recall, and f1-score achieved by the classifier over the k-fold cross validation.
It also stores plots of the graphs used for training as shown in figures 5 and 6 of the paper for problem formulation 1 and 2 respectively.
