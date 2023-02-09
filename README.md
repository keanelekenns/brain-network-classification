TODO: UPDATE README



In the “experiments” directory of this repository, two files named matlab sdp.npy and python sdp.npy can be found.
These files contain example outputs of the Matlab and Python SDP solvers respectively for the same given input.
It was found that the outputted values differed only on the order of 10E−6 to 10E−4, which is likely due to a difference in precision between the libraries used.

test_local_search_omission.py
The last step of the local search algorithm described
in section 3.1.3 appeared to be redundant, and during the translation of the code,
this step was accidentally omitted. A test was later carried out in which 5 difference
networks were each used with 1000 random node sets in the local search algorithm.
Over the 5000 iterations of the algorithm, the complement of the improved node-set
was never superior, hence it was very unlikely it could have affected the results of the
replication.

percentile_alphas.ipynb
Alpha values were obtained from the entire group of brain networks within each category based on the best percentiles reported by Lanciano et al.