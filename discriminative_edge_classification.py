import numpy as np
import argparse
import utils
import classification

def tune_num_edges(graphs, labels):
    best_accuracy = 0
    num_edges = 0
    for i in range(1, 31):
        print("Trying num_edges =", i, end="... ")
        new_accuracy = classification.classify(graphs, labels, discriminative_edges_graphs_to_points, num_folds=5, supress_output=True, num_edges=i)
        print("accuracy =", new_accuracy)
        if new_accuracy > best_accuracy:
            best_accuracy = new_accuracy
            num_edges = i
    return num_edges

def discriminative_edges_graphs_to_points(train_graphs, train_labels, test_graphs, a_label, b_label, num_edges):
    
    # Create and Write Summary Graphs
    summary_A = utils.summary_graph(train_graphs[np.where(train_labels == a_label)])
    summary_B = utils.summary_graph(train_graphs[np.where(train_labels == b_label)])
        
    # Get the difference network between the edge weights in group A and B
    # Note that (u,v) is the same as (v,u), so we extract the upper triangle of the matrix
    diff_net = np.triu(summary_A - summary_B, k=1)

    # Find the num_edges most positive and most negative edge diffs
    partitions = np.argpartition(diff_net, (num_edges, -num_edges), axis=None)
    top_n = np.unravel_index(partitions[-num_edges:], diff_net.shape)
    bottom_n = np.unravel_index(partitions[:num_edges], diff_net.shape)

    def graph_to_point(graph):
        graph[np.where(graph==0)] = -1
        return np.array([np.dot(diff_net[top_n], graph[top_n]),
                         np.dot(diff_net[bottom_n], graph[bottom_n]),
                         np.sum(np.multiply(graph, diff_net))])

    train_points = np.array(list(map(graph_to_point, train_graphs)))
    test_points = np.array(list(map(graph_to_point, test_graphs)))

    return train_points, test_points

def main():
    parser = argparse.ArgumentParser(description='Graph Classification using Discriminative Edges')
    parser.add_argument('A_dir', help='Filepath to class A directory containing brain network files.', type=str)
    parser.add_argument('B_dir', help='Filepath to class B directory containing brain network files.', type=str)
    parser.add_argument('a_label', help='Label for class A', type=str, default="A")
    parser.add_argument('b_label', help='Label for class B', type=str, default="B")
    parser.add_argument('-n','--num-edges', help='Number of positive and negative edges to use for classification.', type=int)
    parser.add_argument('-k','--num-folds', help='Number of times to fold data in k-fold cross validation (default: 5).', type=int, default = 5)
    parser.add_argument('-loo', '--leave-one-out', help='If present, perform leave-one-out cross validation (can be computationally expensive). This will cause num-folds to be ignored.', default=False, action="store_true")
    parser.add_argument('-pre','--plot-prefix', help='A string to prepend to plot names. If present, plots will be generated in the ./plots/ directory. Otherwise, no plots will be generated', type=str, default="")
    parser.add_argument('-t', '--tune', help='Whether or not to tune the number of edges before running the cross-validation (increases runtime).\
                        Note that tuning automatically occurs if the number of edges are not provided.', default=False, action="store_true")

    args = parser.parse_args()

    # Read brain graph files into numpy arrays
    graphs_A = utils.get_graphs_from_files(args.A_dir)
    graphs_B = utils.get_graphs_from_files(args.B_dir)

    graphs, labels = utils.label_and_concatenate_graphs(graphs_A, graphs_B, a_label=args.a_label, b_label=args.b_label)

    num_edges = args.num_edges
    if not num_edges or args.tune:
        print("Tuning number of edges...")
        num_edges = tune_num_edges(graphs, labels)

    print("\nPerforming Discriminitive Edge Classification on Brain Networks")

    print("Number of Edges: ", num_edges)
    classification.classify(graphs, labels, discriminative_edges_graphs_to_points,
                            args.num_folds, args.leave_one_out, args.plot_prefix,
                            random_state=23, num_edges=num_edges)
    

if __name__ == "__main__":
    main()