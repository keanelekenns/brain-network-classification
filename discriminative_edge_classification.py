from typing import ValuesView
import numpy as np
import argparse
import utils
import dense_subgraph
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

def graphs_to_points(graphs, diff_net, top_edge_indices, bottom_edge_indices, top_sum=None, bottom_sum=None):
    """
    Inputs:
        graphs - A 3D numpy array representing brain graphs.
    Returns:
        points - A 2D numpy array representing graph coordinates according to the discriminative edge technique.
    """
    if not top_sum or not bottom_sum:
        top_sum = np.sum(diff_net[top_edge_indices])
        bottom_sum = np.sum(diff_net[bottom_edge_indices])
    
    return np.array(list(map(lambda graph:
                                np.array([np.dot(diff_net[top_edge_indices], graph[top_edge_indices])/top_sum,
                                          np.dot(diff_net[bottom_edge_indices], graph[bottom_edge_indices])/bottom_sum]),
                                    graphs)))

def classify(graphs, labels, num_edges, num_folds=5,
             prefix="", disable_plotting=False):
    # Variables used for reporting at the end

    # Cumulative confusion matrix is used to report on classifier metrics over all of the k folds.
    cumulative_confusion_matrix = np.zeros((2,2))
    # Keep track of the edges that are chosen in all of the k folds.
    important_edges = []

    # k-fold cross validation
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=23)
    i = 0
    for train_index, test_index in skf.split(graphs, labels):
        train_graphs, test_graphs = graphs[train_index], graphs[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
        # Create and Write Summary Graphs
        summary_A = utils.summary_graph(train_graphs[np.where(train_labels == utils.A_LABEL)])
        summary_B = utils.summary_graph(train_graphs[np.where(train_labels == utils.B_LABEL)])
        
        classifier = LinearSVC(random_state=23)

        # Get the difference network between the edge weights in group A and B
        # Note that (u,v) is the same as (v,u), so we extract the upper triangle of the matrix
        diff_net = np.triu(summary_A - summary_B, k=1)

        # Find the num_edges most positive and most negative edge diffs
        partitions = np.argpartition(diff_net, (num_edges, -num_edges), axis=None)
        top_n = np.unravel_index(partitions[-num_edges:], diff_net.shape)
        bottom_n = np.unravel_index(partitions[:num_edges], diff_net.shape)
        top_sum = np.sum(diff_net[top_n])
        bottom_sum = np.sum(diff_net[bottom_n])

        print(top_n)
        if not important_edges: #Check if list is empty
            important_edges = [(set(top_n[0]),set(top_n[1])),
                                (set(bottom_n[0]),set(bottom_n[1]))]
        else:
            important_edges = [(set(top_n[0]).intersection(important_edges[0][0]),
                                set(top_n[1]).intersection(important_edges[0][1])),
                               (set(bottom_n[0]).intersection(important_edges[1][0]),
                                set(bottom_n[1]).intersection(important_edges[1][1]))]
        print("IMPORTANT EDGES", important_edges)

        points = graphs_to_points(np.concatenate((train_graphs, test_graphs)), diff_net, top_n, bottom_n, top_sum, bottom_sum)
        points = StandardScaler().fit_transform(points)
        train_points = points[:train_graphs.shape[0]]
        test_points = points[train_graphs.shape[0]:]

        classifier.fit(train_points, train_labels)
        test_pred = classifier.predict(test_points)
        if(not disable_plotting):
            utils.plot_points(train_points, train_labels,
                        "plots/{}DE-{}-train".format(prefix,i))
            utils.plot_points(test_points, test_pred,
                        "plots/{}DE-{}-test-pred".format(prefix,i))
            utils.plot_points(test_points, test_labels,
                        "plots/{}DE-{}-test-true".format(prefix,i))

        # print(classification_report(test_labels, test_pred))
        # print(confusion_matrix(test_labels, test_pred))
        # print(evaluate_classifier(confusion_matrix(test_labels, test_pred)))
        # metrics += evaluate_classifier(confusion_matrix(test_labels, test_pred))
        cumulative_confusion_matrix += confusion_matrix(test_labels, test_pred)
        i += 1

    print("\nMetrics using cumulative confusion matrix:")
    print(cumulative_confusion_matrix)
    print("Accuracy: {}\nPrecision: {}\nRecall: {}\nF1: {}"
            .format(*utils.evaluate_classifier(cumulative_confusion_matrix)))

    print("\nImportant Nodes: ", important_edges)

def main():
    parser = argparse.ArgumentParser(description='Graph Classification using Discriminative Edges')
    parser.add_argument('A_dir', help='Filepath to class A directory containing brain network files.', type=str)
    parser.add_argument('B_dir', help='Filepath to class B directory containing brain network files.', type=str)
    parser.add_argument('-n','--num-edges', help='Number of positive and negative edges to use for classification (default: 25).', type=int, default = 25)
    parser.add_argument('-k','--num-folds', help='Number of times to fold data in k-fold cross validation (default: 5).', type=int, default = 5)
    parser.add_argument('-dp', '--disable-plotting', help='If present, plots will NOT be generated in the ./plots/ directory.', default=False, action="store_true")
    parser.add_argument('-pre','--prefix', help='A string to prepend to plot names.', type=str, default="")

    args = parser.parse_args()
    
    # Read brain graph files into numpy arrays
    graphs_A = utils.get_graphs_from_files(args.A_dir)
    graphs_B = utils.get_graphs_from_files(args.B_dir)

    graphs, labels = utils.get_AB_labels(graphs_A, graphs_B)

    print("\nPerforming {}-fold cross validation...".format(args.num_folds))
    classify(graphs, labels, args.num_edges, args.num_folds,
             args.prefix, args.disable_plotting)
    

if __name__ == "__main__":
    main()