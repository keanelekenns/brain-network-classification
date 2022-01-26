import argparse
import os
from functools import reduce
import numpy as np

parser = argparse.ArgumentParser(description='Graph Classification via Contrast Subgraph')

parser.add_argument('d', help='dataset', type=str)
parser.add_argument('a', help='Group A', type=str)
parser.add_argument('b', help='Group B', type=str)
parser.add_argument('alpha', help='alpha', type=float)

args = parser.parse_args()

dir1 = "datasets/{}/{}/".format(args.d, args.a)
c1 = ["{}{}".format(dir1, filename) for filename in os.listdir(dir1)]


dir2 = "datasets/{}/{}/".format(args.d, args.b)
c2 = ["{}{}".format(dir2, filename) for filename in os.listdir(dir2)]


# Create and Write Summary Graphs
summary_c1 = reduce(lambda x,y:x+y, map(lambda file: np.loadtxt(file), c1))/len(os.listdir(dir1))
summary_c2 = reduce(lambda x,y:x+y, map(lambda file: np.loadtxt(file), c2))/len(os.listdir(dir2))

diff_net = summary_c1 - summary_c2

edge_diffs = np.sort(diff_net.flatten())
print(edge_diffs[0], edge_diffs[-1])