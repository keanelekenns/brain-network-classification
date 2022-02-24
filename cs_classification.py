import numpy as np
import argparse
import utils
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

A_LABEL = "A"
B_LABEL = "B"
N_SPLITS = 5

def evaluate_classifier(confusion_matrix):
    """
    Inputs:
        confusion_matrix - returned by confusion_matrix in sklearn
    Returns:
        Accuracy, Precision, Recall, F1 - Classifier metrics as defined by 
        https://towardsdatascience.com/classification-performance-metrics-69c69ab03f17
    """
    TP = confusion_matrix[0,0]
    TN = confusion_matrix[1,1]
    FP = confusion_matrix[0,1]
    FN = confusion_matrix[1,0]

    accuracy = (TP + TN)/(TP + TN + FP + FN)
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    f1 = 2*precision*recall/(precision + recall)

    return accuracy, precision, recall, f1

def cs_p1_graphs_to_points(graphs, cs_a_b, cs_b_a):
    """
    Uses the CS-P1 method for creating data points from brain graphs
    Inputs:
        graphs - A 3D numpy array representing brain graphs.
        cs_a_b - A 1D numpy array representing the contrast subgraph generated from
        finding a dense subgraph in the difference graph summary_A - summary_B.
        cs_b_a - Similar to cs_a_b, only the difference graph used to generate it was
        summary_B - summary_A.
    Returns:
        points - A 2D numpy array representing graph coordinates according to the CS-P1 formulation.
    """
    return np.array(list(map(lambda graph:
                                np.array([utils.contrast_subgraph_overlap(graph, cs_a_b),
                                          utils.contrast_subgraph_overlap(graph, cs_b_a)]),
                             graphs)))

def cs_p2_graphs_to_points(graphs, contrast_subgraph, summary_A, summary_B):
    """
    Uses the CS-P2 method for creating data points from brain graphs
    Inputs:
        graphs - A 3D numpy array representing brain graphs.
        contrast_subgraph - A 1D numpy array representing the contrast subgraph generated from
        finding a dense subgraph in the absolute difference graph abs(summary_A - summary_B).
        summary_A - A 2D numpy array with the shape of a brain graph, where each entry is the
        percentage of graphs in class A that contain that given edge.
        summary_B - A 2D numpy array with the shape of a brain graph, where each entry is the
        percentage of graphs in class B that contain that given edge.
    Returns:
        points - A 2D numpy array representing graph coordinates according to the CS-P2 formulation.
    """
    return np.array(list(map(lambda graph:
                                np.array([
                                    utils.l1_norm(utils.induce_subgraph(graph, contrast_subgraph),
                                                utils.induce_subgraph(summary_B, contrast_subgraph)),
                                    utils.l1_norm(utils.induce_subgraph(graph, contrast_subgraph),
                                                utils.induce_subgraph(summary_A, contrast_subgraph))]),
                             graphs)))

def plot_cs_p1(cs_a_b, cs_b_a, graphs_A, graphs_B):

    induced_by_a_b = np.zeros(graphs_A.shape[0])
    induced_by_b_a = np.zeros(graphs_A.shape[0])
    i = 0
    for graph in graphs_A:
        induced_by_a_b[i] = utils.contrast_subgraph_overlap(graph, cs_a_b)
        induced_by_b_a[i] = utils.contrast_subgraph_overlap(graph, cs_b_a)
        i +=1

    fig, ax = plt.subplots()
    ax.scatter(induced_by_a_b, induced_by_b_a, c="#5a7bfc")

    induced_by_a_b = np.zeros(graphs_B.shape[0])
    induced_by_b_a = np.zeros(graphs_B.shape[0])
    i = 0
    for graph in graphs_B:
        induced_by_a_b[i] = utils.contrast_subgraph_overlap(graph, cs_a_b)
        induced_by_b_a[i] = utils.contrast_subgraph_overlap(graph, cs_b_a)
        i +=1
    
    ax.scatter(induced_by_a_b, induced_by_b_a, c="#fcaa1b")
    plt.savefig("plot.jpg")

def get_AB_labels(graphs_A, graphs_B):
    """
    Merge graph arrays together and return corresponding labels
    Inputs:
        graphs_A - A 3D numpy array representing a group of brain graphs in class A.
        graphs_B - A 3D numpy array representing a group of brain graphs in class B.
    Returns:
        graphs - A 3D numpy array representing a group of brain graphs in classes A and B.
        labels - A 1D numpy array holding class labels for each graph in graphs.
    """
    labels_A = [A_LABEL]*len(graphs_A)
    labels_B = [B_LABEL]*len(graphs_B)
    graphs = np.concatenate((graphs_A, graphs_B))
    labels = np.array(labels_A + labels_B)
    return graphs, labels

def main():
    parser = argparse.ArgumentParser(description='Graph Classification via Contrast Subgraphs')
    parser.add_argument('A_dir', help='Filepath to class A directory containing brain network files', type=str)
    parser.add_argument('B_dir', help='Filepath to class B directory containing brain network files', type=str)
    parser.add_argument('alpha', help='Penalty value for contrast subgraph size (varies from 0 to 1)', type=float)
    parser.add_argument('-p', help='Problem Forumlation (default: 1)', type=int, default = 1, choices={1,2})

    args = parser.parse_args()
    if(args.alpha < 0 or args.alpha > 1):
        raise ValueError("alpha should be between 0 and 1 inclusive.")
    
    # Read brain graph files into numpy arrays
    graphs_A = utils.get_graphs_from_files(args.A_dir)
    graphs_B = utils.get_graphs_from_files(args.B_dir)

    graphs, labels = get_AB_labels(graphs_A, graphs_B)

    metrics = np.zeros(4) # 4 metrics: accuracy, precision, recall, f1
    # 5-fold cross validation
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=23)
    for train_index, test_index in skf.split(graphs, labels):
        train_graphs, test_graphs = graphs[train_index], graphs[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        # Create and Write Summary Graphs
        summary_A = utils.summary_graph(train_graphs[np.where(train_labels == A_LABEL)])
        summary_B = utils.summary_graph(train_graphs[np.where(train_labels == B_LABEL)])
        
        classifier = LinearSVC()

        # Get the difference network between the edge weights in group A and B
        if args.p == 1:
            diff_a_b = summary_A - summary_B
            diff_b_a = summary_B - summary_A
            # Call function(s) for generating contrast subgraphs

            # TEMPORARILY HARDCODED
            # Children TD-ASD alpha = 0.8
            cs_a_b = np.array([4, 6, 8, 9, 13, 15, 92, 88, 60])
            # Children ASD-TD alpha = 0.8
            cs_b_a = np.array([36, 37, 71, 41, 74, 76, 77, 79, 81, 55, 38, 95])

            classifier.fit(cs_p1_graphs_to_points(train_graphs, cs_a_b, cs_b_a), train_labels)
            test_pred = classifier.predict(cs_p1_graphs_to_points(test_graphs, cs_a_b, cs_b_a))
        else:
            diff = abs(summary_A - summary_B)
            # Call function(s) for generating contrast subgraph

            # TEMPORARILY HARDCODED
            cs = np.array([4, 6, 8, 9, 13, 15, 92, 88, 60])

            classifier.fit(cs_p2_graphs_to_points(train_graphs, cs, summary_A, summary_B), train_labels)
            test_pred = classifier.predict(cs_p2_graphs_to_points(test_graphs, cs, summary_A, summary_B))

        print(classification_report(test_labels, test_pred))
        print(confusion_matrix(test_labels, test_pred))
        print(evaluate_classifier(confusion_matrix(test_labels, test_pred)))
        metrics += evaluate_classifier(confusion_matrix(test_labels, test_pred))

    metrics /= N_SPLITS
    print("Average Metrics:")
    print("Accuracy: ", metrics[0])
    print("Precision: ", metrics[1])
    print("Recall: ", metrics[2])
    print("F1: ", metrics[3])

if __name__ == "__main__":
    main()