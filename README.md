# contrast-subgraph
This repository contains a replication of "Explainable Classification of Brain Networks via Contrast Subgraphs" by Lanciano et al. (referred to as "the original authors" from here on). The repository for their paper (referred to as "the paper" from here on) can be found [here](https://github.com/tlancian/contrast-subgraph).

Significant modifications have been made to the original authors' code. For example, this repository contains a program named cs_classification.py for evaluating the classifier described in the paper.

Additionally, a new method was attempted in extreme_edge_classification.py, but was found to be ineffective.

## Usage

### cs_classification.py
```
usage: cs_classification.py [-h] [-a a] [-a2 a] [--tune-alpha] [-p {1,2}] [-k NUM_FOLDS] [-pre PREFIX] [-s {sdp,qp}] A_dir B_dir

Graph Classification via Contrast Subgraphs

positional arguments:
  A_dir                 Filepath to class A directory containing brain network files
  B_dir                 Filepath to class B directory containing brain network files

optional arguments:
  -h, --help            show this help message and exit
  -a a, --alpha a       Penalty value for contrast subgraph size (varies from 0 to 1). If not provided, --tune-alpha is set to true.
  -a2 a, --alpha2 a     A secondary alpha value to use for the contrast subgraph from B to A (only applies if problem formulation is 1). Note that the original alpha is used for both contrast subgraphs if this is not provided.
  --tune-alpha          Whether or not to tune the alpha hyperparameter(s) before running the cross-validation (increases runtime). Note that alpha is automatically tuned if no alpha value is provided.
  -p {1,2}, --problem {1,2}
                        Problem Formulation (default: 1)
  -k NUM_FOLDS, --num-folds NUM_FOLDS
                        Number of times to fold data in k-fold cross validation (default: 5)
  -pre PREFIX, --prefix PREFIX
                        A string to prepend to plot names
  -s {sdp,qp}, --solver {sdp,qp}
                        Solver to use for finding a contrast subgraph (default: sdp)
```

The output of this program is a report on the average accuracy, precision, recall, and f1-score achieved by the classifier over the k-fold cross validation. If you would like to store this report, simply append ```> report.txt``` to the command and it will be stored in that file (the git repo ignores files matching report*.txt).

cs_classification.py also generates plots of the graphs used for training as shown in figures 5 and 6 of the paper for problem formulation 1 and 2 respectively, and stores them in the plots directory. Additionally, if the alpha parameter(s) were tuned, the convergence of the minimization algorithm is plotted and stored in convergence.png in the plots directory.

