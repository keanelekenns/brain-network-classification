import numpy as np
import argparse
import utils
import classification

SUMMARY_GRAPH_EDGE_WEIGHT_THRESHOLD = 0.1

# TRIANGLE_RATIO_THRESHOLD = 0.8
TRIANGLE_RATIO_DIFF_THRESHOLD = 0.2

def count_triangles_in_graph(triangles, graph):
    count = 0
    for x,y,z in triangles:
        if graph[x][y] == 1 and graph[x][z] == 1 and graph[y][z] == 1:
            count += 1
    return count

def triangle_histogram(graphs, triangles):
    counts = [0]*len(triangles)
    for graph in graphs:
        for i in range(len(triangles)):
            x,y,z = triangles[i]
            if graph[x][y] == 1 and graph[x][z] == 1 and graph[y][z] == 1:
                counts[i] += 1
    return np.array(counts)

def get_triangles(graph):
    nodes = np.arange(graph.shape[0])
    graph = np.triu(graph, k=1)
    triangles = []
    for i in range(nodes.shape[0]):
        for j in range(i + 1, nodes.shape[0]):
            if graph[i][j] == 1:
                neighbours = graph[i] + graph[j]
                triangles += [(i,j,k) for k in nodes[np.where(neighbours==2)]]
    return np.array(triangles)

def triangles_graphs_to_points(train_graphs, train_labels, test_graphs):
    # Create and Write Summary Graphs
    graphs_a = train_graphs[np.where(train_labels == utils.A_LABEL)]
    graphs_b = train_graphs[np.where(train_labels == utils.B_LABEL)]
    summary_A = utils.summary_graph(graphs_a)
    summary_B = utils.summary_graph(graphs_b)

    diff_net = summary_A - summary_B
    diff_net = np.triu(diff_net, k=1)

    diff_net_a = diff_net.copy()
    diff_net_a[diff_net_a > SUMMARY_GRAPH_EDGE_WEIGHT_THRESHOLD] = 1
    diff_net_a[diff_net_a <= SUMMARY_GRAPH_EDGE_WEIGHT_THRESHOLD] = 0

    diff_net_b = diff_net.copy()
    diff_net_b[diff_net_b >= -SUMMARY_GRAPH_EDGE_WEIGHT_THRESHOLD] = 0
    diff_net_b[diff_net_b < -SUMMARY_GRAPH_EDGE_WEIGHT_THRESHOLD] = 1

    triangles_a = get_triangles(diff_net_a)
    triangles_b = get_triangles(diff_net_b)

    a_in_a = triangle_histogram(graphs_a, triangles_a)/len(graphs_a)
    b_in_a = triangle_histogram(graphs_a, triangles_b)/len(graphs_a)
    a_in_b = triangle_histogram(graphs_b, triangles_a)/len(graphs_b)
    b_in_b = triangle_histogram(graphs_b, triangles_b)/len(graphs_b)

    important_triangles_a = triangles_a[np.where(a_in_a - a_in_b > TRIANGLE_RATIO_DIFF_THRESHOLD)]
    important_triangles_b = triangles_b[np.where(b_in_b - b_in_a > TRIANGLE_RATIO_DIFF_THRESHOLD)]

    print(f"Important A: {important_triangles_a}\nImportant B: {important_triangles_b}")

    train_points = np.array(list(map(lambda graph:
                                np.array([count_triangles_in_graph(important_triangles_a, graph),
                                          count_triangles_in_graph(important_triangles_b, graph)]),
                             train_graphs)))

    test_points = np.array(list(map(lambda graph:
                                np.array([count_triangles_in_graph(important_triangles_a, graph),
                                          count_triangles_in_graph(important_triangles_b, graph)]),
                             test_graphs)))
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