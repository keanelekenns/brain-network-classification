import numpy as np
import argparse
import utils
import dense_subgraph
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

A_LABEL = "A"
B_LABEL = "B"

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

def plot_points(points, labels, plotname):
    x_vals = points[:,0]
    y_vals = points[:,1]
    x_vals_A, y_vals_A = x_vals[np.where(labels == A_LABEL)], y_vals[np.where(labels == A_LABEL)]
    x_vals_B, y_vals_B = x_vals[np.where(labels == B_LABEL)], y_vals[np.where(labels == B_LABEL)]

    fig, ax = plt.subplots()
    ax.scatter(x_vals_A, y_vals_A, c="#5a7bfc")
    ax.scatter(x_vals_B, y_vals_B, c="#fcaa1b")
    plt.savefig(plotname)

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
    parser.add_argument('-k', help='Number of times to fold data in k-fold cross validation (default: 5)', type=int, default = 5)
    parser.add_argument('-a', help='A secondary alpha value to use for the contrast subgraph from B to A \
            (only applies if problem formulation is 1). Note that the original alpha is used for both contrast subgraphs \
            if this is not provided.', type=float)
    parser.add_argument('-prefix', help='A string to prepend to plot names', type=str, default="")
    parser.add_argument('-solver', help='Solver to use for finding a contrast subgraph (default: sdp)', type=str, default = "sdp", choices={"sdp","qp"})

    args = parser.parse_args()
    if(args.alpha < 0 or args.alpha > 1):
        raise ValueError("alpha should be between 0 and 1 inclusive.")
    if(args.a and (args.a < 0 or args.a > 1)):
        raise ValueError("secondary alpha should be between 0 and 1 inclusive.")
    
    if args.solver == "sdp":
        solver = dense_subgraph.sdp
    elif args.solver == "qp":
        solver = dense_subgraph.qp
    
    # Read brain graph files into numpy arrays
    graphs_A = utils.get_graphs_from_files(args.A_dir)
    graphs_B = utils.get_graphs_from_files(args.B_dir)

    graphs, labels = get_AB_labels(graphs_A, graphs_B)
    # 4 metrics: accuracy, precision, recall, f1
    metrics = np.zeros(4)
    # Keep track of the nodes that are common to all contrast subgraphs found
    important_nodes = []

    # k-fold cross validation
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=23)
    i = 0
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

            cs_a_b = solver(diff_a_b, args.alpha)
            cs_b_a = solver(diff_b_a, args.a if args.a else args.alpha)
            print("CONTRAST SUBGRAPHS\n",cs_a_b, cs_b_a)
            if not important_nodes: #Check if list is empty
                important_nodes = [set(cs_a_b), set(cs_b_a)]
            else:
                important_nodes = [set(cs_a_b).intersection(important_nodes[0]),
                                   set(cs_b_a).intersection(important_nodes[1])]
            print("IMPORTANT NODES", important_nodes)
            plot_points(cs_p1_graphs_to_points(train_graphs, cs_a_b, cs_b_a),
                        train_labels,
                        "plots/{}CS-P1-{}".format(args.prefix,i))
            classifier.fit(cs_p1_graphs_to_points(train_graphs, cs_a_b, cs_b_a), train_labels)
            test_pred = classifier.predict(cs_p1_graphs_to_points(test_graphs, cs_a_b, cs_b_a))
        else:
            diff = abs(summary_A - summary_B)
            
            cs = solver(diff, args.alpha)
            if not important_nodes: #Check if list is empty
                important_nodes = set(cs)
            else:
                important_nodes = set(cs).intersection(important_nodes)
            print("IMPORTANT NODES", important_nodes)
            plot_points(cs_p2_graphs_to_points(train_graphs, cs, summary_A, summary_B),
                        train_labels,
                        "plots/{}CS-P2-{}".format(args.prefix,i))
            classifier.fit(cs_p2_graphs_to_points(train_graphs, cs, summary_A, summary_B), train_labels)
            test_pred = classifier.predict(cs_p2_graphs_to_points(test_graphs, cs, summary_A, summary_B))

        print(classification_report(test_labels, test_pred))
        print(confusion_matrix(test_labels, test_pred))
        print(evaluate_classifier(confusion_matrix(test_labels, test_pred)))
        metrics += evaluate_classifier(confusion_matrix(test_labels, test_pred))
        i += 1
    metrics /= args.k
    print("Average Metrics:")
    print("Accuracy: ", metrics[0])
    print("Precision: ", metrics[1])
    print("Recall: ", metrics[2])
    print("F1: ", metrics[3])
    print("Important Nodes: ", important_nodes)

if __name__ == "__main__":
    main()