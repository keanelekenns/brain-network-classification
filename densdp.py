# coding: utf-8
import networkx as nx
import oqc_sdp
import numpy as np
import scipy.io
# from new_greedy_oqc import localSearchNegativeOQC
import cvxpy as cp

def densdp(diff_net, alpha):
    """
    Find a contrast subgraph given a differenct network (effectively a densely weighted subgraph)
    Inputs:
        diff_net: A 2D numpy array of shape (|V|, |V|) where V is the vertex
        set for the studied graphs. The value of diff_net[i,j] is the weight
        of the edge from node i to node j.
        alpha - Penalty value for large graphs (between 0 and 1).
    Returns:
        constrast_subgraph - A 1D numpy array containing vertex indexes of a contrast subgraph.
    """
    G = nx.from_numpy_array(diff_net)
    print ("Loaded graph with %s nodes and %s edges" % (len(G), G.number_of_edges()))
    w, d = oqc_sdp._make_coefficient_matrices(diff_net)
    P = np.matrix(w - alpha * d)

    n = len(P)
    X = cp.Variable((n,n), symmetric=True)
    constraints = [X >> 0]
    constraints += [cp.diag(X) == 1]
    P_4 = P / 4
    prob = cp.Problem(cp.Maximize(cp.trace(P_4 @ X)),
                      constraints)
    prob.solve()

    L = oqc_sdp.semidefinite_cholesky(X)
    nodeset, obj, obj_rounded = oqc_sdp.random_projection_qp(L, P, diff_net, alpha, t=1000)
    nodes = list(G.nodes())
    S_bar = G.subgraph([nodes[i - 1] for i in nodeset])
    print([nodes[i - 1] for i in nodeset])
    # do local search to try to improve solution
    # print(nx.cliques_containing_node(G, S_bar))
    # S, obj_rounded = localSearchNegativeOQC(G, alpha, t_max=50, seed=S_bar)
    return [nodes[i - 1] for i in nodeset]
