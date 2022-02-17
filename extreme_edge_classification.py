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
top_sum = np.sum(diff_net[top_n])
bottom_sum = np.sum(diff_net[bottom_n])

# Note: A = 1, B = 0
predictions_A = np.zeros(len(test_A))
i = 0
for file in test_A:
    g = np.loadtxt(file)
    a_similarity = np.dot(diff_net[top_n], g[top_n])/top_sum
    b_similarity = np.dot(diff_net[bottom_n], g[bottom_n])/bottom_sum
    print("A: {}, B: {}".format(a_similarity, b_similarity))
    if a_similarity > b_similarity:
        predictions_A[i] = 1
    i += 1

predictions_B = np.zeros(len(test_B))
i = 0
for file in test_B:
    g = np.loadtxt(file)
    a_similarity = np.dot(diff_net[top_n], g[top_n])/top_sum
    b_similarity = np.dot(diff_net[bottom_n], g[bottom_n])/bottom_sum
    if a_similarity > b_similarity:
        predictions_B[i] = 1
    i += 1

TP = np.sum(predictions_A)
FN = len(test_A) - TP

FP = np.sum(predictions_B)
TN = len(test_B) - FP

Accuracy = (TP + TN)/(TP + TN + FP + FN)
Precision = TP/(TP + FP)
Recall = TP/(TP + FN)
F1 = 2*Precision*Recall/(Precision + Recall)

print(predictions_A)
print(predictions_B)
print("Accuracy: {}".format(Accuracy))
print("Precision: {}".format(Precision))
print("Recall: {}".format(Recall))
print("F1: {}".format(F1))