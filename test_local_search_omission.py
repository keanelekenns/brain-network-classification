import numpy as np
import utils
import random
from sklearn.model_selection import StratifiedKFold

def localSearch_Tsourakakis_with_last_step(graph, node_set, alpha, max_iterations=100):
    """
    The Tsourakakis implementation of localSearch (with modification from Cadena).
    Given a weighted graph and a set of nodes, attempt to build a locally
    optimal set of nodes (S) such that they maximize the function:
    f_alpha(S) = SUM(graph[u,v] - alpha) for all u,v in S
    Inputs:
        graph - A 2D numpy array of shape (|V|, |V|) where V is the vertex
        set for the graph. The value of graph[i,j] is the weight
        of the edge from node i to node j.
        node_set - A 1D numpy array containing vertex indexes of a dense subgraph
        of graph (to be optimized).
        alpha - Penalty value for large graphs (between 0 and 1).
        max_iterations - Maximum number of times to refine the outputted node set.
    Returns:
        refined_node_set - A 1D numpy array containing vertex indexes
        of a dense subgraph of graph.
    """
    # print("Contrast subgraph before local search", node_set)
    # Don't want to double count edges, so only take upper triangle
    g = np.triu(graph, k=1)
    nodes = np.arange(graph.shape[0])
    # Create masks for nodes inside the refined_node_set and those outside it
    S = np.array([v in node_set for v in nodes])
    V = ~S
    edge_weight_surplus = utils.edge_weight_surplus(g, nodes[S], alpha)
    i = 0
    while i < max_iterations:
        i += 1
        while True:
            found = False
            for node in nodes[V]:
                S[node] = True # Put node in S
                new_edge_weight_surplus = utils.edge_weight_surplus(g, nodes[S], alpha)
                if new_edge_weight_surplus > edge_weight_surplus:
                    # print(nodes[S], "Old", edge_weight_surplus, "New", new_edge_weight_surplus)
                    edge_weight_surplus = new_edge_weight_surplus
                    V[node] = False # Take node out of V
                    found = True
                else:
                    S[node] = False # Take node back out of S
            if not found:
                break

        found = False
        for node in nodes[S]:
            S[node] = False # Take node out of S
            new_edge_weight_surplus = utils.edge_weight_surplus(g, nodes[S], alpha)
            if new_edge_weight_surplus >= edge_weight_surplus:
                # print(nodes[S], "Old", edge_weight_surplus, "New", new_edge_weight_surplus)
                edge_weight_surplus = new_edge_weight_surplus
                V[node] = True # Put node in V
                found = True
                break
            else:
                S[node] = True # Put node back in S
        if not found:
            break
        if i == max_iterations:
            print("LocalSearch reached maximum number of iterations")
    # print("Contrast subgraph after {} iterations of local search".format(i), nodes[S])

    if utils.edge_weight_surplus(g, nodes[V], alpha) > edge_weight_surplus:
        print("Would have used compliment")
    
    return nodes[S].copy()

random.seed(a=42)

DATASET_NAME = "male"

GRAPH_DIR_PREFIX = "./data/lanciano_datasets_corr_thresh_80/"
DATA_DESCRIPTOR = "Lanciano-Processed"

A_GRAPH_DIR = f"{GRAPH_DIR_PREFIX}{DATASET_NAME}/asd/"
B_GRAPH_DIR = f"{GRAPH_DIR_PREFIX}{DATASET_NAME}/td/"

a_label="ASD"
b_label="TD"

graphs_A = utils.get_graphs_from_files(A_GRAPH_DIR)
graphs_B = utils.get_graphs_from_files(B_GRAPH_DIR)

graphs, labels = utils.label_and_concatenate_graphs(graphs_A=graphs_A, graphs_B=graphs_B, a_label=a_label, b_label=b_label)

nodes = list(np.arange(graphs[0].shape[0]))
print(f"{len(nodes)} nodes")

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in folds.split(graphs, labels):
    train_graphs = graphs[train_index]
    train_labels = labels[train_index]

    # This is the algorithm used by DiscriminativeEdgesTransformer in the fit function

    # Create and Write Summary Graphs
    # Note that (u,v) is the same as (v,u), so we extract the upper triangle of the matrices
    summary_A = np.triu(utils.summary_graph(graphs[np.where(labels == a_label)]), k=1)
    summary_B = np.triu(utils.summary_graph(graphs[np.where(labels == b_label)]), k=1)
        
    # Get the difference network between the edge weights in group A and B
    diff_net = summary_A - summary_B
    for i in range(1000):
        node_set = random.sample(population=nodes, k=random.randrange(1,len(nodes)))
        localSearch_Tsourakakis_with_last_step(diff_net, node_set, 0)




