import argparse
import os
from functools import reduce
import numpy as np
import subprocess
import networkx as nx
import cvxpy as cp

def parse_args():
	parser = argparse.ArgumentParser(description='Graph Classification via Contrast Subgraph')

	parser.add_argument('d', help='dataset', type=str)
	parser.add_argument('a', help='Group A', type=str)
	parser.add_argument('b', help='Group B', type=str)
	parser.add_argument('alpha', help='alpha', type=float)
	parser.add_argument('-p', help='Problem Forumlation (default: 1)', default = "1", choices=["1", "2"])

	args = parser.parse_args()

	dir1 = "datasets/{}/{}/".format(args.d,args.a)
	c1 = ["{}{}".format(dir1,elem) for elem in os.listdir(dir1)]


	dir2 = "datasets/{}/{}/".format(args.d,args.b)
	c2 = ["{}{}".format(dir2,elem) for elem in os.listdir(dir2)]
	return dir1, dir2, c1, c2, args.p, args.alpha

def summary_graph(dir1, dir2, c1, c2, p):
	summary_c1 = reduce(lambda x,y:x+y,map(lambda x: np.loadtxt(x, delimiter = " "),c1))/len(os.listdir(dir1))
	summary_c2 = reduce(lambda x,y:x+y,map(lambda x: np.loadtxt(x, delimiter = " "),c2))/len(os.listdir(dir2))

	if p == "1":
	    diff_net = summary_c1 - summary_c2
	elif p == "2":
	    diff_net = abs(summary_c1 - summary_c2)
	return diff_net

def densdp(diff_net, alpha):
    A = diff_net
    G = nx.from_numpy_array(A)
    print ("Loaded graph with %s nodes and %s edges" % (len(G), G.number_of_edges()))
    w, d = _make_coefficient_matrices(A)
    P = np.matrix(w - alpha * d)

    n = len(P)
    X = cp.Variable((n,n), symmetric=True)
    constraints = [X >> 0]
    constraints += [cp.diag(X) == 1]
    P_4 = P / 4
    prob = cp.Problem(cp.Maximize(cp.trace(P_4 @ X)),
                      constraints)
    prob.solve()

    L = semidefinite_cholesky(X)
    nodeset, obj, obj_rounded = random_projection_qp(L, P, A, alpha, t=1000)
    nodes = list(G.nodes())
    S_bar = G.subgraph([nodes[i - 1] for i in nodeset])
    print([nodes[i - 1] for i in nodeset])

def _make_coefficient_matrices(A, weight='weight'):
    N = len(A)
    # w is the matrix of coefficients for 
    # the objective function
    w = np.zeros((N + 1, N + 1))
    # the (0, j)th entry of w is the sum of column j
    w_0j = np.zeros(N + 1)
    w_0j[1:] = A.sum(axis=0)
    w[0, :] = w_0j
    # the (i, 0)th entry of w is the sum of row i
    w_i0 = np.zeros(N + 1)
    w_i0[1:] = A.sum(axis=1).T
    w[:, 0] = w_i0
    # the rest of w (i.e., entries (i, j), such that i != 0 and j != 0)
    # are just the adjacency matrix of G
    w[1:, 1:] = A
    # diagonal elements should be all zero (no self-loops)
    np.fill_diagonal(w, 0.)
    
    # d is the matrix of coefficients for constraint (1)
    # For constraint (1), we want each edge in the final solution to
    # contribute a weight of 1
    # We can think of d as being the corresponding matrix w
    # for a complete graph (i.e., each edge has weight 1)
    d = np.ones((N + 1, N + 1))
    # the (0, j)th entry of d is the sum of column j (i.e., N - 1)
    d_0j = np.ones(N + 1) * (N - 1)
    d[0, :] = d_0j
    # same for the (i, 0)th entries
    d_i0 = np.ones(N + 1) * (N - 1)
    d[:, 0] = d_i0
    np.fill_diagonal(d, 0.)

    return w, d

def semidefinite_cholesky(X):
    # the Cholesky decomposition is defined for 
    # positive definite matrices. We have to add
    # a small constant to X to make it PD
    V = np.array(X.value) if type(X) not in [np.array, np.ndarray] else X

    eps = 1e-10
    while True:
        try:
            L = np.linalg.cholesky(V + (eps * np.identity(len(V))))
            L = np.matrix(L)
            break
        except np.linalg.LinAlgError:
            eps *= 10
    # print a warning if epsilon starts getting too big
    if (eps >= 1e-3):
        print ("WARNING in Cholesky Decomposition:")
        print ("Input matrix had to be perturbed by", eps)
    return L

def random_projection_qp(L, P, A, alpha, t=100, seed=None, return_x_rounded=False):
    '''
        Input:
        L: Solution matrix from SDP
        P: ceofficient matrix of SDP
        A: Adjacency matrix
        alpha: parameter of OQC problem
        
        Returns:
        S: Set of nodes obtained from rounding
        obj_orig: The objective value before rounding
        obj: The objective value of the rounded matrix
    '''
    # random projection algorithm
    # Repeat t times
    eps = 1e-6
    count = 0
    sum_weights = A.sum() - alpha * (len(A) * (len(A) - 1))
    # initial solution: S = \emptyset (1, -1, ... , -1)
    x_rounded = -1 * np.ones(len(L))
    x_rounded[0] = 1
    obj = 0
    if seed is not None:
        x_rounded[seed] = 1
        obj = ((sum_weights + np.matrix(x_rounded) * P * np.matrix(x_rounded).T) / 8.)[0, 0]
    obj_orig = (sum_weights + np.trace(P * (L * L.T))) / 8.

    while (count < t):
        r = np.matrix(np.random.normal(size=len(L)))
        L_0_sign = np.sign(L[0] * r.T)[0, 0]
        x = np.sign(L * r.T) == L_0_sign
        x = x * 1
        x[x == 0] = -1
        o = ((sum_weights + x.T * P * x) / 8.)[0, 0]
        #print "number of nodes in set:",  x[x == 1].shape
        #S = G.subgraph([(n - 1) for n in xrange(1, len(L)) if x[n] == x[0]])
        #print o
        #print x.shape
        if o > obj + eps:
            x_rounded = x
            obj = o
            #print "found a better solution"
            #print obj
            #S = [(n - 1) for n in xrange(1, len(L)) if x_rounded[n] == x_rounded[0]]
            #print S
        count += 1
    # solution is the set of nodes with the same orientation
    # as x_0
    S = [n for n in range(1, len(L)) if x_rounded[n] == x_rounded[0]]
    if return_x_rounded:
        x_rounded = np.matrix(x_rounded)
        if x_rounded.shape[0] != len(L):
            x_rounded = x_rounded.T
        return x_rounded
    return S, obj_orig, obj

if __name__ == '__main__':
	dir1, dir2, c1, c2, p, alpha = parse_args()
	diff_net = summary_graph(dir1, dir2, c1, c2, p)
	densdp(diff_net, alpha)