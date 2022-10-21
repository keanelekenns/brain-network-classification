# coding: utf-8
from datetime import datetime
import cadena
import numpy as np
import cvxpy as cp
import cvxopt
import utils

def localSearch_Tsourakakis(graph, node_set, alpha, max_iterations=100):
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

def local_search_enns(graph, node_set, alpha, max_iterations=10):
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
    g = np.triu(graph, k=1)
    nodes = np.arange(graph.shape[0])
    # Create masks for nodes inside the refined_node_set and those outside it
    S = np.array([v in node_set for v in nodes])
    # If there is nothing in the initial node_set, the algorithm won't be able to add nodes
    # So start it off with two nodes that have the heaviest edge between them
    if not S.any():
        u, v = np.unravel_index(np.argmax(graph), graph.shape)
        S[u] = True
        S[v] = True
        
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
    # start_time = datetime.now()

    w, d = cadena._make_coefficient_matrices(diff_net)
    P = np.matrix(w - alpha * d)

    n = len(P)
    X = cp.Variable((n,n), symmetric=True)
    constraints = [X >> 0]
    constraints += [cp.diag(X) == 1]
    P_4 = P / 4
    prob = cp.Problem(cp.Maximize(cp.trace(P_4 @ X)),
                      constraints)
    prob.solve()
    # np.save("python_sdp.npy", X.value) # Used to compare to original code

    L = cadena.semidefinite_cholesky(X)
    node_set, obj_orig, obj_rounded = cadena.random_projection_qp(L, P, diff_net, alpha, t=1000)

    # finish_sdp_time = datetime.now()
    # print(f"Time for SDP: {finish_sdp_time - start_time}")

    # print(f"CS before local search: {node_set}")
    # objective_value_sdp = utils.edge_weight_surplus(graph=diff_net, node_set=np.array(node_set), alpha=alpha)
    # print(f"Objective function value: {objective_value_sdp}")

    cs = localSearch_Tsourakakis(diff_net, node_set, alpha)

    # print(f"CS after local search: {cs}")
    # objective_value_local_search = utils.edge_weight_surplus(graph=diff_net, node_set=cs, alpha=alpha)
    # print(f"Objective function value: {objective_value_local_search}")

    # if objective_value_sdp > 0:
    #     print(f"Local Search improved solution by {((objective_value_local_search - objective_value_sdp)/objective_value_sdp)*100}%")

    # finish_time = datetime.now()
    # print(f"Time for local search: {finish_time - finish_sdp_time}")
    # print(f"Time to find CS: {finish_time - start_time}")
    return cs

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
    # start_time = datetime.now()
    
    N = diff_net.shape[0]
    assert(N == diff_net.shape[1])

    # The objective function to maximize would be diff_net - alpha, but our qp solver
    # can only minimize an expression, so we swap the signs.
    objective_function = alpha - diff_net
    np.fill_diagonal(objective_function, 0)

    # CVXOPT can only minimize x in the expression (1/2) x.T @ P @ x + q.T @ x
    # subject to Gx << h (and Ax = b, but that's not applicable here)
    P = objective_function / 2
    q = np.sum(objective_function, axis=0) / 2
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
    node_set = np.where(x > 0)[0]

    # finish_qp_time = datetime.now()
    # print(f"Time for QP: {finish_qp_time - start_time}")
    
    # print(f"CS before local search: {node_set}")
    # objective_value_qp = utils.edge_weight_surplus(graph=diff_net, node_set=node_set, alpha=alpha)
    # print(f"Objective function value: {objective_value_qp}")

    cs = local_search_enns(diff_net, node_set, alpha)

    # print(f"CS after local search: {cs}")
    # objective_value_local_search = utils.edge_weight_surplus(graph=diff_net, node_set=cs, alpha=alpha)
    # print(f"Objective function value: {objective_value_local_search}")

    # if objective_value_qp > 0:
    #     print(f"Local Search improved solution by {((objective_value_local_search - objective_value_qp)/objective_value_qp)*100}%")

    # finish_time = datetime.now()
    # print(f"Time for local search: {finish_time - finish_qp_time}")
    # print(f"Time to find CS: {finish_time - start_time}")
    return cs