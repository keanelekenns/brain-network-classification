import numpy as np
import argparse
import utils
import classification

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

def discriminative_edges_graphs_to_points(train_graphs, train_labels, test_graphs, num_edges, important_edges=[]):
    
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
    top_sum = np.sum(diff_net[top_n])
    bottom_sum = np.sum(diff_net[bottom_n])


    if not important_edges: #Check if list is empty
        important_edges = [(set(top_n[0]),set(top_n[1])),
                            (set(bottom_n[0]),set(bottom_n[1]))]
    else:
        important_edges = [(set(top_n[0]).intersection(important_edges[0][0]),
                            set(top_n[1]).intersection(important_edges[0][1])),
                            (set(bottom_n[0]).intersection(important_edges[1][0]),
                            set(bottom_n[1]).intersection(important_edges[1][1]))]

    train_points = graphs_to_points(train_graphs, diff_net, top_n, bottom_n, top_sum, bottom_sum)
    test_points = graphs_to_points(test_graphs, diff_net, top_n, bottom_n, top_sum, bottom_sum)

    return train_points, test_points

def main():
    parser = argparse.ArgumentParser(description='Graph Classification using Discriminative Edges')
    parser.add_argument('A_dir', help='Filepath to class A directory containing brain network files.', type=str)
    parser.add_argument('B_dir', help='Filepath to class B directory containing brain network files.', type=str)
    parser.add_argument('-n','--num-edges', help='Number of positive and negative edges to use for classification (default: 25).', type=int, default = 25)
    parser.add_argument('-k','--num-folds', help='Number of times to fold data in k-fold cross validation (default: 5).', type=int, default = 5)
    parser.add_argument('-loo', '--leave-one-out', help='If present, perform leave-one-out cross validation (can be computationally expensive). This will cause num-folds to be ignored.', default=False, action="store_true")
    parser.add_argument('-p', '--plot', help='If present, plots will be generated in the ./plots/ directory.', default=False, action="store_true")
    parser.add_argument('-pre','--plot-prefix', help='A string to prepend to plot names.', type=str, default="")

    args = parser.parse_args()

    print("Performing Discriminitive Edge Classification on Brain Networks\n")
    
    # Read brain graph files into numpy arrays
    graphs_A = utils.get_graphs_from_files(args.A_dir)
    graphs_B = utils.get_graphs_from_files(args.B_dir)

    graphs, labels = utils.get_AB_labels(graphs_A, graphs_B)

    num_folds = args.num_folds
    if args.leave_one_out:
        num_folds = len(labels)

    classification.classify(graphs, labels, num_folds,
                            discriminative_edges_graphs_to_points,
                            args.plot, args.plot_prefix,
                            random_state=23, num_edges=args.num_edges)
    

if __name__ == "__main__":
    main()