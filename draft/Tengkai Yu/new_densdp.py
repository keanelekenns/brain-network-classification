# coding: utf-8
import networkx as nx
import new_oqc_sdp
import sys
import os
import numpy as np
import scipy.io
import subprocess
import time
from new_greedy_oqc import localSearchNegativeOQC
import pickle
import cvxpy as cp

DIR = 'tmp'
CVXPATH = 'cvx'
alpha = float(sys.argv[2])
def make_matlab_script(prefix, N):
    with open("%s/solveSDP%s.m" % (CVXPATH, prefix.replace('-', '')), 'w') as f:
        f.write("cvx_setup\n")
        f.write("load('../%s/%s.mat')\n" % (DIR, prefix))
        f.write("cvx_solver sedumi\n")
        f.write("cvx_begin sdp\n")
        f.write("    variable X(%s, %s) symmetric\n" % (N, N))
        f.write("    maximize trace(P*X)\n")
        f.write("    subject to\n")
        f.write("        X >= 0;\n")
        f.write("        diag(X) == ones(%s, 1);\n" % N)
        f.write("cvx_end\n")
        f.write("save('../%s/%s.txt', 'X', '-ASCII');\n" % (DIR, prefix))
        f.write("exit;\n")


def write_output(sdp_results, outfname, subgraphfile):
    S, obj, obj_rounded = sdp_results

    print ("Returning subgraph with OQC score", obj_rounded, "(%s)" % obj)
    n = len(S)
    if 'weight' in S.edges_iter(data=True).next()[2]:
        e = sum(data['weight'] for u, v, data in S.edges_iter(data=True))
    else:
        e = S.number_of_edges()
    header = "|S|,|E|,density,diameter,triangle density,OQC,obj\n"
    with open(outfname, 'w') as f:
        f.write(header)
        if n > 0:
            f.write(str(n) + ',')
            f.write(str(S.number_of_edges()) + ',')
            if n > 1:
                f.write(str(2. * e / (n * (n - 1))) + ',')
            else:
                f.write(str(0) + ',')
            if nx.is_connected(S):
                f.write(str(nx.diameter(S)) + ',')
            else:
                f.write('inf,')
            if n > 2:
                f.write(str(2. * sum(i for i in nx.triangles(S).itervalues()) / (n * (n - 1) * (n - 2))) + ',')
            else:
                f.write(str(0) + ',')
            f.write(str(obj_rounded) + ',')
            f.write(str(obj) + '\n')
        else:
            f.write("0,0,0,0,0,0,0,")
            f.write("%s\n" % obj)
    nx.write_weighted_edgelist(S, subgraphfile)


def main():
    graphfile = sys.argv[1]
    #outfname = sys.argv[2]
    #subgraphfile = sys.argv[3]
    upper_bound = 0

    if not os.path.exists(DIR):
        os.mkdir(DIR)
    # load graph
    A = np.load(graphfile)
    G = nx.from_numpy_array(A)
    print ("Loaded graph with %s nodes and %s edges from %s" % (len(G), G.number_of_edges(), graphfile))
    w, d = new_oqc_sdp._make_coefficient_matrices(A)
    P = np.matrix(w - alpha * d)
    filename = os.path.split(graphfile)[1]
    # make matlab input
    prefix = os.path.splitext(filename)[0]
    scipy.io.savemat('%s/%s.mat' % (DIR, prefix), mdict={'P': (1 / 4.) * P})

    n = len(P)
    X = cp.Variable((n,n), symmetric=True)
    constraints = [X >> 0]
    constraints += [cp.diag(X) == 1]
    P_4 = P / 4
    prob = cp.Problem(cp.Maximize(cp.trace(P_4 @ X)),
                      constraints)
    prob.solve()

    L = new_oqc_sdp.semidefinite_cholesky(X)
    nodeset, obj, obj_rounded = new_oqc_sdp.random_projection_qp(L, P, A, alpha, t=1000)
    nodes = list(G.nodes())
    S_bar = G.subgraph([nodes[i - 1] for i in nodeset])
    print([nodes[i - 1] for i in nodeset])
    # do local search to try to improve solution
    # print(nx.cliques_containing_node(G, S_bar))
    # S, obj_rounded = localSearchNegativeOQC(G, alpha, t_max=50, seed=S_bar)

    
    # with open('%s/%s.txt' % (DIR, prefix), 'w') as res:
    #     res.write(str(S.nodes()))
    #     res.close()
        
    



if __name__ == '__main__':
    main()
