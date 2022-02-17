import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import utils

parser = argparse.ArgumentParser(description='Graph Classification via Extreme Edges')

parser.add_argument('A_dir', help='Filepath to class A directory containing brain network files', type=str)
parser.add_argument('B_dir', help='Filepath to class B directory containing brain network files', type=str)
parser.add_argument('e', help='The number of edges used for discriminating between classes', type=int)

args = parser.parse_args()

graphs_A = utils.get_graphs_from_files(args.A_dir)
graphs_B = utils.get_graphs_from_files(args.B_dir)

# Split into training and testing samples
train_A, test_A = train_test_split(graphs_A, test_size=0.2, random_state=23)
train_B, test_B = train_test_split(graphs_B, test_size=0.2, random_state=23)

# Create and Write Summary Graphs
summary_A = utils.summary_graph(train_A)
summary_B = utils.summary_graph(train_B)

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
for g in test_A:
    a_similarity = np.dot(diff_net[top_n], g[top_n])/top_sum
    b_similarity = np.dot(diff_net[bottom_n], g[bottom_n])/bottom_sum
    print("A: {}, B: {}".format(a_similarity, b_similarity))
    if a_similarity > b_similarity:
        predictions_A[i] = 1
    i += 1

predictions_B = np.zeros(len(test_B))
i = 0
for g in test_B:
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