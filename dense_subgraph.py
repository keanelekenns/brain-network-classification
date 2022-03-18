# coding: utf-8
import networkx as nx
import oqc_sdp
import numpy as np
import cvxpy as cp
import cvxopt
import utils

def localSearch_Tsourakakis(graph, node_set, alpha, max_iterations=10):
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
    return nodes[S].copy()

def localSearch(graph, node_set, alpha, max_iterations=10):
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
    # print("Contrast subgraph before local search", node_set)
    # Don't want to double count edges, so only take upper triangle
    g = np.triu(graph, k=1)
    nodes = np.arange(graph.shape[0])
    # Create masks for nodes inside the refined_node_set and those outside it
    S = np.array([v in node_set for v in nodes])
    V = ~S
    edge_weight_surplus = utils.edge_weight_surplus(g, nodes[S], alpha)
    changes_made = True
    i = 0
    while changes_made and i < max_iterations:
        changes_made = False
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
                    changes_made = True
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
                    changes_made = True
                else:
                    S[node] = True # Put node back in S
            if not found:
                break
        if i == max_iterations:
            print("LocalSearch reached maximum number of iterations")
    # print("Contrast subgraph after {} iterations of local search".format(i), nodes[S])
    return nodes[S].copy()
            

def sdp(diff_net, alpha):
    """
    Find a contrast subgraph by finding a dense subgraph using an SDP solver.
    Inputs:
        diff_net: A 2D numpy array of shape (|V|, |V|) where V is the vertex
        set for the studied graphs. The value of diff_net[i,j] is the weight
        of the edge from node i to node j.
        alpha - Penalty value for large graphs (between 0 and 1).
    Returns:
        constrast_subgraph - A 1D numpy array containing vertex indexes of a contrast subgraph.
    """
    G = nx.from_numpy_array(diff_net)
    # print ("Loaded graph with %s nodes and %s edges" % (len(G), G.number_of_edges()))
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
    return localSearch(diff_net, nodeset, alpha)

def qp(diff_net, alpha):
    """
    Find a contrast subgraph by finding a dense subgraph using a QP solver.
    Inputs:
        diff_net: A 2D numpy array of shape (|V|, |V|) where V is the vertex
        set for the studied graphs. The value of diff_net[i,j] is the weight
        of the edge from node i to node j.
        alpha - Penalty value for large graphs (between 0 and 1).
    Returns:
        constrast_subgraph - A 1D numpy array containing vertex indexes of a contrast subgraph.
    """
    N = diff_net.shape[0]
    assert(N == diff_net.shape[1])
    objective_function = diff_net - alpha
    np.fill_diagonal(objective_function, 0)

    # CVXOPT can only minimize x in the expression (1/2) x.T @ P @ x + q.T @ x
    # subject to Gx << h (and Ax = b, but that's not applicable here)
    # So we need to invert the objective function, because we want to maximize it.
    P = np.triu(-objective_function, k=1)
    q = np.sum(-objective_function, axis=0)
    G = np.zeros((2*N, N))
    for i in range(N):
        G[2*i,i] = 1
        G[2*i+1,i] = -1
    h = np.ones(2*N)

    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)
    # suppress output
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P,q,G,h)
    x = np.array(sol['x'])
    # objective_value = sol['primal objective']
    nodeset = np.where(x > 0)[0]
    
    return localSearch(diff_net, nodeset, alpha)