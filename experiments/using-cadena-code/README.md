These alpha values were obtained from the entire group of brain networks within each category based on the best percentiles reported by Lanciano et al. (see percentile_alphas.ipynb in the root directory).
The alpha values were then fed into the original code in Lanciano et al.'s repository (https://github.com/tlancian/contrast-subgraph), which also uses all of the brain networks of each category to find contrast subgraphs (contrast subgraphs are listed in replication.xlsx).
The contrast subgraphs obtained were then used to translate brain networks into 2D vectors in 5 Fold Cross Validation.
With little tuning, 79% accuracy was easily obtained.
However, this is completely invalid, as the method involves the test data when constructing the prediction model.
This experiment was only conducted to try to explain Lanciano et al.'s high accuracy reportings.