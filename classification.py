import numpy as np


def get_contrast_subgraph_overlap(subject_brain, contrast_subgraphs ):
    """
    Inputs:
        subject_brain - A numpy array representing a subject's brain network as an adjacency graph.
        The shape should be (|V|, |V|) where V is the set of nodes in the brain network. The value
        at subject_brain[u,v] should be 1 if the edge (u,v) is in the graph, and 0 otherwise. Note
        that the graph should be undirected, and thus, only the upper right triangle of the matrix
        will be considered. The graph should not contain self loops (i.e. there must be zeros on the
        diagonal).
        constrast_subgraphs - A list of 1D numpy arrays representing contrast subgraphs and
        containing vertex indexes of each subgraph.
    Returns:
        category_counts - A numpy array with one entry per contrast subgraph. Each entry corresponds
        to the number of edges in subject_brain having both endpoints in the associated contrast
        subgraph.
    """
    brain_network = np.triu(subject_brain)
    category_counts = np.zeros(len(contrast_subgraphs))
    i = 0
    for subgraph in contrast_subgraphs:
        for source in subgraph:
            brain_dests = np.nonzero(brain_network[source,:])
            category_counts[i] += np.count_nonzero(np.isin(subgraph, brain_dests, assume_unique=True))
        i += 1
    return category_counts



# def main():


# if __name__ == "__main__":
#     main()