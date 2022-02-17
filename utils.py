import numpy as np
import os

def contrast_subgraph_overlap(subject_brain, contrast_subgraph):
    """
    Inputs:
        subject_brain - A numpy array representing a subject's brain graph as an adjacency graph.
        The shape should be (|V|, |V|) where V is the set of nodes in the brain graph. The value
        at subject_brain[u,v] should be 1 if the edge (u,v) is in the graph, and 0 otherwise. Note
        that the graph should be undirected, and thus, only the upper right triangle of the matrix
        will be considered. The graph should not contain self loops (i.e. there must be zeros on the
        diagonal).
        constrast_subgraph - A 1D numpy array containing vertex indexes of a contrast subgraph
    Returns:
        overlap_count - The number of edges in subject_brain having both endpoints in the
        contrast subgraph.
    """
    brain_graph = np.triu(subject_brain)
    overlap_count = 0
    for source in contrast_subgraph:
        brain_dests = np.nonzero(brain_graph[source,:])
        overlap_count += np.count_nonzero(np.isin(contrast_subgraph, brain_dests, assume_unique=True))
    return overlap_count

def l1_norm(g1, g2):
    """
    Compute the L1 norm between two graphs with equal vertex sets.
    Inputs:
        g1, g2 - 2D numpy arrays representing graphs. They should have the same shape
        of (|V|, |V|) where V is their vertex set. The value at index [u,v] should be
        the weight of edge (u,v) in the respective graph.
    Returns:
        norm - The sum of the absolute value of g1 - g2.
    """
    return np.sum(np.absolute(np.triu(g1 - g2)))

def induce_subgraph(graph, nodes):
    """
    Inputs:
        graph - A 2D numpy array representing a graph.
        nodes - A 1D numpy array representing a subset of the graph's node set.
    Returns:
        subgraph - A 2D numpy array of shape (len(nodes), len(nodes)) representing
        the subgraph of the input graph induced by nodes.
    """
    return np.copy(graph[np.ix_(nodes, nodes)])

def get_graphs_from_files(dir):
    """
    Inputs:
        dir - Path to directory containing graph files. These files are assumed to
        be .txt files with a square 2D array of 0s and 1s.
    Returns:
        graphs - A 3D numpy array of shape (num_graphs, |V|, |V|) where V is the vertex
        set for all graphs. Can be used as input to summary_graph().
    """
    return np.array([np.loadtxt("{}{}".format(dir, filename)) for filename in os.listdir(dir)])

def summary_graph(graphs):
    """
    Inputs:
        graphs - A 3D numpy array of shape (num_graphs, |V|, |V|) where V is the vertex
        set for all graphs. Same as the output of get_graphs_from_files().
    Returns:
        summary - A 2D numpy array of shape (|V|, |V|) where V is the vertex
        set for all graphs. Assuming each graph has entries of 0 or 1, the summary graph
        will contain, as entries, the percentage of graphs that each edge is in.
    """
    return np.sum(graphs, axis=0)/graphs.shape[0]