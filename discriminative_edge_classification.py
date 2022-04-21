import numpy as np
import argparse
import utils
import classification

def discriminative_edges_graphs_to_points(train_graphs, train_labels, test_graphs, num_edges):
    
    # Create and Write Summary Graphs
    summary_A = utils.summary_graph(train_graphs[np.where(train_labels == utils.A_LABEL)])
    summary_B = utils.summary_graph(train_graphs[np.where(train_labels == utils.B_LABEL)])
        
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
    parser.add_argument('-n','--num-edges', help='Number of positive and negative edges to use for classification (default: 25).', type=int, default = 25)
    parser.add_argument('-k','--num-folds', help='Number of times to fold data in k-fold cross validation (default: 5).', type=int, default = 5)
    parser.add_argument('-loo', '--leave-one-out', help='If present, perform leave-one-out cross validation (can be computationally expensive). This will cause num-folds to be ignored.', default=False, action="store_true")
    parser.add_argument('-pre','--plot-prefix', help='A string to prepend to plot names. If present, plots will be generated in the ./plots/ directory. Otherwise, no plots will be generated', type=str, default="")

    args = parser.parse_args()

    print("Performing Discriminitive Edge Classification on Brain Networks\n")
    
    # Read brain graph files into numpy arrays
    graphs_A = utils.get_graphs_from_files(args.A_dir)
    graphs_B = utils.get_graphs_from_files(args.B_dir)

    graphs, labels = utils.get_AB_labels(graphs_A, graphs_B)

    classification.classify(graphs, labels, discriminative_edges_graphs_to_points,
                            args.num_folds, args.leave_one_out, args.plot_prefix,
                            random_state=23, num_edges=args.num_edges)
    

if __name__ == "__main__":
    main()