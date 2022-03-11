# coding: utf-8
import networkx as nx
import oqc_sdp
import numpy as np
import cvxpy as cp
import utils

def localSearch(graph, node_set, alpha, max_iterations=2):
    """
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
    # Don't want to double count edges, so only take upper triangle
    g = np.triu(graph)
    nodes = np.arange(graph.shape[0])
    # Create masks for nodes inside the refined_node_set and those outside it
    S = np.array([v in node_set for v in nodes])
    V = ~S
    edge_weight_surplus = utils.edge_weight_surplus(g, nodes[S], alpha)
    found = False
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
        while True:
            found = False
            for node in nodes[S]:
                S[node] = False # Take node out of S
                new_edge_weight_surplus = utils.edge_weight_surplus(g, nodes[S], alpha)
                if new_edge_weight_surplus >= edge_weight_surplus:
                    # print(nodes[S], "Old", edge_weight_surplus, "New", new_edge_weight_surplus)
                    edge_weight_surplus = new_edge_weight_surplus
                    V[node] = True # Put node in V
                    found = True
                else:
                    S[node] = True # Put node back in S
            if not found:
                break
    return nodes[S].copy()
            

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
    print("NODESET", nodeset)
    return localSearch(diff_net, nodeset, alpha)
