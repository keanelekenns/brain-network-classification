import argparse
import os
from functools import reduce
import numpy as np
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Graph Classification via Contrast Subgraph')

parser.add_argument('d', help='dataset', type=str)
parser.add_argument('a', help='Group A', type=str)
parser.add_argument('b', help='Group B', type=str)
parser.add_argument('e', help='The number of edges used for discriminating between classes', type=int)

args = parser.parse_args()

dir_A = "datasets/{}/{}/".format(args.d, args.a)
files_A = ["{}{}".format(dir_A, filename) for filename in os.listdir(dir_A)]

dir_B = "datasets/{}/{}/".format(args.d, args.b)
files_B = ["{}{}".format(dir_B, filename) for filename in os.listdir(dir_B)]

# Split into training and testing samples
train_A, test_A = train_test_split(files_A, test_size=0.2, random_state=23)
train_B, test_B = train_test_split(files_B, test_size=0.2, random_state=23)

# Create and Write Summary Graphs
summary_A = reduce(lambda x,y:x+y, map(lambda file: np.loadtxt(file), train_A))/len(train_A)
summary_B = reduce(lambda x,y:x+y, map(lambda file: np.loadtxt(file), train_B))/len(train_B)

# Get the difference network between the edge weights in group A and B
# Note that (u,v) is the same as (v,u), so we extract the upper triangle of the matrix
diff_net = np.triu(summary_A - summary_B, 1)

# Find the e highest and lowest edge diffs
partitions = np.argpartition(diff_net, (args.e, -args.e), axis=None)
top_n = np.unravel_index(partitions[-args.e:], diff_net.shape)
bottom_n = np.unravel_index(partitions[:args.e], diff_net.shape)
print(bottom_n, top_n, diff_net[bottom_n], diff_net[top_n])