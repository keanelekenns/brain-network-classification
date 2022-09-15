import numpy as np
import networkx as nx
import argparse
import utils
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

def graphs_to_points(graphs, cliques_a, cliques_b):
    """
    Inputs:
        graphs - A 3D numpy array representing brain graphs.
    Returns:
        points - A 2D numpy array representing graph coordinates according to the clique technique.
    """
    def graph_to_point(graph, cliques_a, cliques_b):
        g = nx.from_numpy_array(np.triu(graph, k=1))
        cliques = [x for x in nx.enumerate_all_cliques(g) if len(x)==3]
        a = len([x for x in cliques if x in cliques_a])
        b = len([x for x in cliques if x in cliques_b])
        return np.array([a,b])
    
    return np.array(list(map(lambda graph: graph_to_point(graph, cliques_a, cliques_b), graphs)))

def classify(graphs, labels, a_label, b_label, num_folds=5,
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
        summary_A = utils.summary_graph(train_graphs[np.where(train_labels == a_label)])
        summary_B = utils.summary_graph(train_graphs[np.where(train_labels == b_label)])
        
        summary_A = np.triu(summary_A, k=1)
        temp_a = np.zeros(summary_A.shape)
        temp_a[np.where(summary_A > 0.9)] = 1
        g_a = nx.from_numpy_array(temp_a)
        cliques_a = [x for x in nx.enumerate_all_cliques(g_a) if len(x)==3]

        summary_B = np.triu(summary_B, k=1)
        temp_b = np.zeros(summary_B.shape)
        temp_b[np.where(summary_B > 0.9)] = 1
        g_b = nx.from_numpy_array(temp_b)
        cliques_b = [x for x in nx.enumerate_all_cliques(g_b) if len(x)==3]

        c_a = [x for x in cliques_a if x not in cliques_b]
        c_b = [x for x in cliques_b if x not in cliques_a]

        classifier = LinearSVC(random_state=23)

        points = graphs_to_points(np.concatenate((train_graphs, test_graphs)), c_a, c_b)
        points = StandardScaler().fit_transform(points)
        train_points = points[:train_graphs.shape[0]]
        test_points = points[train_graphs.shape[0]:]

        classifier.fit(train_points, train_labels)
        test_pred = classifier.predict(test_points)
        if(not disable_plotting):
            utils.plot_points(train_points, train_labels,
                        "plots/{}KC-{}-train".format(prefix,i))
            utils.plot_points(test_points, test_pred,
                        "plots/{}KC-{}-test-pred".format(prefix,i))
            utils.plot_points(test_points, test_labels,
                        "plots/{}KC-{}-test-true".format(prefix,i))

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

    print("\nImportant Edges: ", important_edges)

def main():
    parser = argparse.ArgumentParser(description='Graph Classification using Discriminative Edges')
    parser.add_argument('A_dir', help='Filepath to class A directory containing brain network files.', type=str)
    parser.add_argument('B_dir', help='Filepath to class B directory containing brain network files.', type=str)
    parser.add_argument('a_label', help='Label for class A', type=str, default="A")
    parser.add_argument('b_label', help='Label for class B', type=str, default="B")
    parser.add_argument('-n','--num-edges', help='Number of positive and negative edges to use for classification (default: 25).', type=int, default = 25)
    parser.add_argument('-k','--num-folds', help='Number of times to fold data in k-fold cross validation (default: 5).', type=int, default = 5)
    parser.add_argument('-l', '--learn', help='If present, important edges from previous folds will be intersected with each new set of important edges.', default=False, action="store_true")
    parser.add_argument('-dp', '--disable-plotting', help='If present, plots will NOT be generated in the ./plots/ directory.', default=False, action="store_true")
    parser.add_argument('-pre','--prefix', help='A string to prepend to plot names.', type=str, default="")

    args = parser.parse_args()
    
    # Read brain graph files into numpy arrays
    graphs_A = utils.get_graphs_from_files(args.A_dir)
    graphs_B = utils.get_graphs_from_files(args.B_dir)

    graphs, labels = utils.label_and_concatenate_graphs(graphs_A, graphs_B, a_label=args.a_label, b_label=args.b_label)

    print("\nPerforming {}-fold cross validation...".format(args.num_folds))
    classify(graphs, labels, args.num_folds,
             args.prefix, args.disable_plotting)
    

if __name__ == "__main__":
    main()