import numpy as np
import argparse
import utils
import classification

def calculate_ratios(counts, total):
    return [count/total for count in counts]

def count_triangles(graphs, triangles):
    counts = [0]*len(triangles)
    for graph in graphs:
        for i in range(triangles.shape[0]):
            x,y,z = triangles[i]
            if graph[x][y] == 1 and graph[x][z] == 1 and graph[y][z] == 1:
                counts[i] += 1
    return counts

def get_triangles(graph):
    nodes = np.arange(graph.shape[0])
    graph = np.triu(graph, k=1)
    triangles = []
    for i in range(nodes.shape[0]):
        for j in range(i + 1, nodes.shape[0]):
            if graph[i][j] == 1:
                neighbours = graph[i] + graph[j]
                triangles += [(i,j,k) for k in nodes[np.where(neighbours==2)]]
    return triangles

def triangles_graphs_to_points(train_graphs, train_labels, test_graphs):
    # Create and Write Summary Graphs
    graphs_a = train_graphs[np.where(train_labels == utils.A_LABEL)]
    graphs_b = train_graphs[np.where(train_labels == utils.B_LABEL)]
    summary_A = utils.summary_graph(graphs_a)
    summary_B = utils.summary_graph(graphs_b)

    nodes = np.arange(summary_A.shape[0])
    train_points = np.zeros((train_graphs.shape[0], 2))
    test_points = np.zeros((test_graphs.shape[0], 2))
    diff_net = summary_A - summary_B
    diff_net = np.triu(diff_net, k=1)

    diff_net_a = diff_net.copy()
    diff_net_a[diff_net_a > 0] = 1
    diff_net_a[diff_net_a < 0] = 0

    diff_net_b = diff_net.copy()
    diff_net_b[diff_net_b < 0] = 1
    diff_net_b[diff_net_b > 0] = 0

    triangles_a = get_triangles(diff_net_a)
    triangles_b = get_triangles(diff_net_b)

    a_in_a = calculate_ratios(count_triangles(graphs_a, triangles_a), graphs_a.shape[0])
    b_in_a = calculate_ratios(count_triangles(graphs_a, triangles_b), graphs_a.shape[0])
    a_in_b = calculate_ratios(count_triangles(graphs_b, triangles_a), graphs_b.shape[0])
    b_in_b = calculate_ratios(count_triangles(graphs_b, triangles_b), graphs_b.shape[0])

    

    return train_points, test_points

def main():
    parser = argparse.ArgumentParser(description='Graph Classification via Contrast Subgraphs')
    parser.add_argument('A_dir', help='Filepath to class A directory containing brain network files.', type=str)
    parser.add_argument('B_dir', help='Filepath to class B directory containing brain network files.', type=str)
    parser.add_argument('-k','--num-folds', help='Number of times to fold data in k-fold cross validation (default: 5).', type=int, default = 5)
    parser.add_argument('-loo', '--leave-one-out', help='If present, perform leave-one-out cross validation (can be computationally expensive). This will cause num-folds to be ignored.', default=False, action="store_true")
    parser.add_argument('-pre','--plot-prefix', help='A string to prepend to plot names. If present, plots will be generated in the ./plots/ directory. Otherwise, no plots will be generated', type=str, default="")

    args = parser.parse_args()
    
    # print("\nPerforming Contrast Subgraph Classification on Brain Networks")
    
    # Read brain graph files into numpy arrays
    graphs_A = utils.get_graphs_from_files(args.A_dir)
    graphs_B = utils.get_graphs_from_files(args.B_dir)

    graphs, labels = utils.get_AB_labels(graphs_A, graphs_B)

    classification.classify(graphs, labels, triangles_graphs_to_points,
                            args.num_folds, args.leave_one_out, args.plot_prefix, random_state=23)
    

if __name__ == "__main__":
    main()