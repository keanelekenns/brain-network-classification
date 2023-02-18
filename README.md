# Setup

This repository uses Poetry for managing dependencies.

To begin, install Poetry version 1.1.14:

```curl -sSL https://install.python-poetry.org | python3 - --version 1.1.14```

Read their [documentation](https://python-poetry.org/docs/) for updated instructions.

To verify Poetry is installed, run ```poetry --version``` the output should be ```Poetry version 1.1.14```

Installing all of the necessary dependencies to a new virtual environment should now be as easy as running ```poetry install``` from the root directory of this repo.

To activate the virtual environment run ```poetry shell``` (if there is an issue, it could be there is a space in the path to the virtual environment).

To run the jupyter notebooks in VS code, you will need to change the kernel to point to the python binary in the virtual environment folder (read Poetry docs to find the location of this). The binary should be in ```path_to_venv/.venv/bin/python```.

---

# Table of Contents

## Directories
- [archive](#archive)
- [data](#data)
- [experiments](#experiments)
- [outputs](#outputs)
- [plots](#plots)
- [scripts](#scripts)

## Files
- [cadena](#cadena)
- [classification](#classification)
- [cs_transformer](#cs_transformer)
- [de_transformer](#de_transformer)
- [dense_subgraph](#dense_subgraph)
- [grid_search_cv](#grid_search_cv)
- [iidaka_transformer](#iidaka_transformer)
- [important_edges](#important_edges)
- [nested_grid_search_cv](#nested_grid_search_cv)
- [percentile_alphas](#percentile_alphas)
- [pipeline](#pipeline)
- [replication](#replication)
- [test_local_search_omission](#test_local_search_omission)
- [utils](#utils)

---

## archive
The files in this directory consitute some of the outdated code that was used during the experiments conducted for this research paper.
Some files were deleted, but could likely be found in the git history if necessary.

To get any of this code to work, it may need to be tweaked. It serves only as an indication as to what experiments were performed, but were either revised or abandoned.

---

## data

This directory contains the data used in this study.

- BrainNetViewer - Contains the files used to create the brain diagrams with the [BrainNetViewer](https://www.nitrc.org/projects/bnv/) program. The edge files were generated with the [important_edges.ipynb](#important_edges) notebook.
- ABIDE - Contains the BOLD timeseries of the ABIDE dataset downloaded from the Preprocessed Connectomes Project amazon s3 bucket and a phenotypic trait file for each of the subjects. The script used to download these files is named download_abide_files.ipynb and can be found in the [scripts](#scripts) directory. Note that the script can download timeseries without GSR and bandpass filtering with a simple modification to one of the variables.
- generated_filt_global - Contains the raw correlation matrices generated in this study using the timeseries_to_networks.ipynb notebook in the [scripts](#scripts) directory.
- lanciano_datasets_corr_thresh_80 - Contains the thresholded correlation matrices provided by Lanciano et al in their project repository.

Each of the directories containing data files have a unique.json file that was generated with the calculate_unique_subjects.py script. This file gives filepaths to each unique subject, since subjects can be found in more than one subdirectory depending on their phenotypic traits.

---

## experiments

This directory contains output files from the experiments reported on in the study.
These output files contain details such as the accuracy, runtime, and confusion matrices of the experiments.

In this directory, two files named matlab sdp.npy and python sdp.npy can be found.
These files contain example outputs of the Matlab and Python SDP solvers respectively for the same given input.
It was found that the outputted values differed only on the order of 10E−6 to 10E−4, which is likely due to a difference in precision between the libraries used.

---

## outputs

The default output directory when running experiments. The gitignore file causes files here to be ignored by the repository.

---

## plots

Similar to the output directory, but contains generated plots. These plots are ignored by git.

---

## scripts

Contains various scripts for downloading and processing files.

---


## cadena

Contains some reused functions from Lanciano et al.'s repository, but they are believed to have originated from the work of Cadena et al.

---

## classification

Functions for performing nested and non-nested cross validation for the experiments, as well as outputting results.


---

## cs_transformer

Transformer class implementing the Contrast Subgraph approach.

---

## de_transformer

Transformer class implementing the Discriminative Edges approach.

---

## dense_subgraph

Functions for finding and refining dense subgraphs. These include the SDP and QP solvers, and the local search algorithms.

---

## grid_search_cv

The notebook for performing the cross validation experiments. Certain lines can be uncommented and commented to run the experiments on different datasets.

---

## iidaka_transformer

Transformer class implementing the Effect Size Thresholding approach.

---

## important_edges

Used to create the edge files for the [BrainNetViewer](#data) directory.
It divides the brain networks into 5 folds and identifies the most discriminative edges by accumulating the difference network values for edges when they are chosen. This is repeated 10 times to get a good idea of which edges are the most important overall.

---

## nested_grid_search_cv

The notebook for performing the nested cross validation experiments. Certain lines can be uncommented and commented to run the experiments on different datasets.

---

## percentile_alphas

This file was used to investigate the hypothesis that Lanciano et al. may have used all of the graphs to find the contrast subgraphs.
Alpha values were obtained from the entire group of brain networks within each category based on the best percentiles reported by Lanciano et al.
These were then used to obtain contrast subgraphs with the original code provided by Lanciano et al.

---

## pipeline

Custom pipeline class similar to the class provided by sklearn, but customized for the purposes of this study.

---

## replication

Notebook for running the replication experiment with the best reported parameters by Lanciano et al.

---

## test_local_search_omission
The last step of the local search algorithm described
in section 3.1.3 appeared to be redundant, and during the translation of the code,
this step was accidentally omitted. A test was later carried out in which 5 difference
networks were each used with 1000 random node sets in the local search algorithm.
Over the 5000 iterations of the algorithm, the complement of the improved node-set
was never superior, hence it was very unlikely it could have affected the results of the
replication.

---

## utils

Various helper functions, such as the logical implementation of using contrast subgraphs to embed the brain networks.

---