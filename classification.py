import numpy as np
import argparse
import utils
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def evaluate_classifier(confusion_matrix):
    """
    Inputs:
        confusion_matrix - returned by confusion_matrix in sklearn
        The labels are as follows: Class A = 0, Class B = 1.
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

def train_cs_p1_classifier(cs_a_b, cs_b_a, train, labels):
    """
    Fit a classifier with the train data using contrast subgraphs
    Inputs:
        cs_a_b - A 1D numpy array representing the contrast subgraph generated from
        finding a dense subgraph in the difference graph summary_A - summary_B.
        cs_b_a - Similar to cs_a_b, only the difference graph used to generate it was
        summary_B - summary_A.
        train - A 3D numpy array representing a group of brain graphs in classes A and B.
        labels - A 1D numpy array representing class labels for each graph in train.
    Returns:
        A trained Linear SVC Classifier from sklearn
    """
    points = np.array(list(map(lambda graph: cs_p1_graph_to_point(graph, cs_a_b, cs_b_a), train)))
    lsvc = LinearSVC()
    lsvc.fit(points, labels)
    return lsvc

def cs_p1(cs_a_b, cs_b_a, test, classifier):
    """
    Uses the CS-P1 method for classifying a test set of brain graphs
    Inputs:
        cs_a_b - A 1D numpy array representing the contrast subgraph generated from
        finding a dense subgraph in the difference graph summary_A - summary_B.
        cs_b_a - Similar to cs_a_b, only the difference graph used to generate it was
        summary_B - summary_A.
        test - A 3D numpy array representing a group of brain graphs in classes A and B.
        classifier - A trained linear SVC classifier from sklearn
    Returns:
        predictions - A 1D numpy array holding prediction values for each graph in test.
        The prediction values are as follows: Class A = 0, Class B = 1.
    """
    points = np.array(list(map(lambda graph: cs_p1_graph_to_point(graph, cs_a_b, cs_b_a), test)))
    return classifier.predict(points)

def cs_p1_graph_to_point(graph, cs_a_b, cs_b_a):
    """
    Uses the CS-P1 method for creating a data point from a brain graph
    Inputs:
        graph - A 2D numpy array representing a brain graph adjacency matrix.
        cs_a_b - A 1D numpy array representing the contrast subgraph generated from
        finding a dense subgraph in the difference graph summary_A - summary_B.
        cs_b_a - Similar to cs_a_b, only the difference graph used to generate it was
        summary_B - summary_A.
    Returns:
        point - A 1D numpy array of length 2 representing a coordinate according to the CS-P1
        formulation.
    """
    return np.array([
        utils.contrast_subgraph_overlap(graph, cs_a_b),
        utils.contrast_subgraph_overlap(graph, cs_b_a)])

def cs_p2_graph_to_point(graph, contrast_subgraph, summary_A, summary_B):
    """
    Uses the CS-P2 method for creating a data point from a brain graph
    Inputs:
        graph - A 2D numpy array representing a brain graph adjacency matrix.
        contrast_subgraph - A 1D numpy array representing the contrast subgraph generated from
        finding a dense subgraph in the absolute difference graph abs(summary_A - summary_B).
        summary_A - A 2D numpy array with the shape of a brain graph, where each entry is the
        percentage of graphs in class A that contain that given edge.
        summary_B - A 2D numpy array with the shape of a brain graph, where each entry is the
        percentage of graphs in class B that contain that given edge.
    Returns:
        point - A 1D numpy array of length 2 representing a coordinate according to the CS-P2
        formulation.
    """
    induced_subgraph = utils.induce_subgraph(graph, contrast_subgraph)
    return np.array([
        utils.l1_norm(induced_subgraph,
                      utils.induce_subgraph(summary_B, contrast_subgraph)),
        utils.l1_norm(induced_subgraph,
                      utils.induce_subgraph(summary_A, contrast_subgraph))
    ])

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
        labels - A 1D numpy array holding class labels for each graph in train.
        The label values are as follows: Class A = 0, Class B = 1.
    """
    labels_A = np.zeros(len(graphs_A))
    labels_B = np.ones(len(graphs_B))
    graphs = np.concatenate((graphs_A, graphs_B))
    labels = np.concatenate((labels_A, labels_B))
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
    # Split into training and testing samples
    train_A, test_A = train_test_split(graphs_A, test_size=0.2, random_state=23)
    train_B, test_B = train_test_split(graphs_B, test_size=0.2, random_state=23)
    # Create and Write Summary Graphs
    summary_A = utils.summary_graph(train_A)
    summary_B = utils.summary_graph(train_B)

    train_graphs, train_labels = get_AB_labels(train_A, train_B)
    test_graphs, test_labels = get_AB_labels(test_A, test_B)

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
    
        classifier = train_cs_p1_classifier(cs_a_b, cs_b_a, train_graphs, train_labels)
        test_pred = cs_p1(cs_a_b, cs_b_a, test_graphs, classifier)
        print(classification_report(test_labels, test_pred, target_names=["A", "B"]))
        print(confusion_matrix(test_labels, test_pred))
        print(evaluate_classifier(confusion_matrix(test_labels, test_pred)))
    else:
        diff = abs(summary_A - summary_B)
        # Call function(s) for generating contrast subgraph

        # TEMPORARILY HARDCODED
        


if __name__ == "__main__":
    main()