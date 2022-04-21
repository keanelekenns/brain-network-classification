import numpy as np
import os
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

A_LABEL = "A"
B_LABEL = "B"

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
    brain_graph = np.triu(subject_brain, k=1)
    overlap_count = 0
    for source in contrast_subgraph:
        brain_dests = np.nonzero(brain_graph[source,:])[0]
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
    return np.sum(np.absolute(np.triu(g1 - g2, k=1)))

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

def edge_weight_surplus(graph, node_set, alpha):
    """
    Given a weighted graph and a set of nodes, calculate the function:
    f_alpha(node_set) = SUM(graph[u,v] - alpha) for all u,v in node_set
    Inputs:
        graph - A 2D numpy array of shape (|V|, |V|) where V is the vertex
        set for the graph. The value of graph[i,j] is the weight
        of the edge from node i to node j.
        node_set - A 1D numpy array containing vertex indexes of a subgraph
        of graph.
        alpha - Penalty value for large graphs (between 0 and 1).
    Returns:
        edge_weight_surplus - A value indicating the edge weight surplus.
        This is given by f_alpha(node_set) above.
    """
    g = np.triu(graph, k=1)
    edge_weight_sum = induce_subgraph(g, node_set).sum()
    N = node_set.shape[0]
    return edge_weight_sum - alpha * (N * (N - 1)) / 2

def evaluate_classifier(confusion_matrix):
    """
    Inputs:
        confusion_matrix - returned by confusion_matrix in sklearn
    Returns:
        Accuracy, Precision, Recall, F1 - Classifier metrics as defined by 
        https://towardsdatascience.com/classification-performance-metrics-69c69ab03f17
    """
    TP = confusion_matrix[0,0]
    TN = confusion_matrix[1,1]
    FP = confusion_matrix[0,1]
    FN = confusion_matrix[1,0]

    accuracy = (TP + TN)/(TP + TN + FP + FN)
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    f1 = 2*precision*recall/(precision + recall)

    return accuracy, precision, recall, f1

def plot_points(points, labels, plotname):
    x_vals = points[:,0]
    a_indices = np.where(labels == A_LABEL)
    b_indices = np.where(labels == B_LABEL)
    if points.shape[1] == 3:
        fig = plt.figure()
        ax = plt.axes(projection ='3d')

        y_vals = points[:,1]
        z_vals = points[:,2]
        x_vals_A, y_vals_A, z_vals_A = x_vals[a_indices], y_vals[a_indices], z_vals[a_indices]
        x_vals_B, y_vals_B, z_vals_B = x_vals[b_indices], y_vals[b_indices], z_vals[b_indices]
        ax.scatter(x_vals_A, y_vals_A, z_vals_A, c="#5a7bfc")
        ax.scatter(x_vals_B, y_vals_B, z_vals_B, c="#fcaa1b")
        plt.savefig(plotname)
        return
    elif points.shape[1] == 1:
        x_vals_A = x_vals[a_indices]
        y_vals_A = np.zeros(x_vals_A.shape)
        x_vals_B = x_vals[b_indices]
        y_vals_B = np.zeros(x_vals_B.shape)
    elif points.shape[1] == 2:
        y_vals = points[:,1]
        x_vals_A, y_vals_A = x_vals[a_indices], y_vals[a_indices]
        x_vals_B, y_vals_B = x_vals[b_indices], y_vals[b_indices]

    fig, ax = plt.subplots()
    ax.scatter(x_vals_A, y_vals_A, c="#5a7bfc")
    ax.scatter(x_vals_B, y_vals_B, c="#fcaa1b")
    plt.savefig(plotname)

# TODO: This function needs work.
def plot_decision_boundary(pred_func, X, y, plotname):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    hx = (x_max - x_min)/100
    hy = (y_max - y_min)/100
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))
    # Predict the function value for the whole grid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.savefig(plotname)

def get_AB_labels(graphs_A, graphs_B):
    """
    Merge graph arrays together and return corresponding labels
    Inputs:
        graphs_A - A 3D numpy array representing a group of brain graphs in class A.
        graphs_B - A 3D numpy array representing a group of brain graphs in class B.
    Returns:
        graphs - A 3D numpy array representing a group of brain graphs in classes A and B.
        labels - A 1D numpy array holding class labels for each graph in graphs.
    """
    labels_A = [A_LABEL]*len(graphs_A)
    labels_B = [B_LABEL]*len(graphs_B)
    graphs = np.concatenate((graphs_A, graphs_B))
    labels = np.array(labels_A + labels_B)
    return graphs, labels

def dsi(points_a, points_b):
    # Based on https://arxiv.org/pdf/2109.05180.pdf

    # Get pairwise euclidian distance between points in same class without repitition
    d_a = np.linalg.norm(points_a[:,None,:] - points_a[None,:,:], axis=-1)
    d_a = d_a[np.triu_indices(d_a.shape[0], k=1)]
    d_b = np.linalg.norm(points_b[:,None,:] - points_b[None,:,:], axis=-1)
    d_b = d_b[np.triu_indices(d_b.shape[0], k=1)]

    # Get pairwise distances between classes
    d_a_b = np.linalg.norm(points_a[:,None,:] - points_b[None,:,:], axis=-1).flatten()

    # Get the KS statistic (distance between samples)
    s_a = ks_2samp(d_a, d_a_b).statistic
    s_b = ks_2samp(d_b, d_a_b).statistic

    # DSI
    return (s_a + s_b)/2