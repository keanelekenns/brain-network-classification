import numpy as np
import argparse
import utils
import dense_subgraph
import classification
from skopt.space import Real
from skopt.utils import use_named_args
from skopt import forest_minimize
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt

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
                                                utils.induce_subgraph(summary_A, contrast_subgraph)),
                                    utils.l1_norm(utils.induce_subgraph(graph, contrast_subgraph),
                                                utils.induce_subgraph(summary_B, contrast_subgraph))]),
                             graphs)))


def tune_alpha_dsi(graphs, labels, initial_alpha=None, initial_alpha2=None,
               problem=1, solver=dense_subgraph.sdp, plot=False):
    graphs_a = graphs[np.where(labels == utils.A_LABEL)]
    graphs_b = graphs[np.where(labels == utils.B_LABEL)]
    summary_A = utils.summary_graph(graphs_a)
    summary_B = utils.summary_graph(graphs_b)
    diff_a_b = summary_A - summary_B
    diff_b_a = summary_B - summary_A
    diff = abs(diff_a_b)

    initial_params = None

    if problem == 1:
        space = [Real(name='alpha', low=0.0, high=1.0),
                 Real(name='alpha2', low=0.0, high=1.0)]
        @use_named_args(space)
        def objective(alpha, alpha2):
            cs_a_b = solver(diff_a_b, alpha)
            cs_b_a = solver(diff_b_a, alpha2)
            return -utils.dsi(cs_p1_graphs_to_points(graphs_a, cs_a_b, cs_b_a),
                        cs_p1_graphs_to_points(graphs_b, cs_a_b, cs_b_a))
        if initial_alpha:
            if initial_alpha2:
                initial_params = [initial_alpha, initial_alpha2]
            else:
                initial_params = [initial_alpha, initial_alpha]
        result = forest_minimize(objective, space, n_calls=20, n_initial_points=5, x0=initial_params, random_state=23)
        if(plot):
            plot_convergence(result)
            plt.savefig("plots/convergence.png")
        return result.x

    elif problem == 2:
        space = [Real(name='alpha', low=0.0, high=1.0)]
        @use_named_args(space)
        def objective(alpha):
            cs = solver(diff, alpha)
            return -utils.dsi(cs_p2_graphs_to_points(graphs_a, cs, summary_A, summary_B),
                        cs_p2_graphs_to_points(graphs_b, cs, summary_A, summary_B))
        
        initial_params = [initial_alpha] if initial_alpha else None
        result = forest_minimize(objective, space, n_calls=20, n_initial_points=5, x0=initial_params, random_state=23)
        if(plot):
            plot_convergence(result)
            plt.savefig("plots/convergence.png")
        return result.x[0]
    else:
        print("Cannot tune alpha - Incorrect value for problem formulation")
        return

def tune_alpha_accuracy(graphs, labels, initial_alpha=None, initial_alpha2=None,
                        problem=1, solver=dense_subgraph.sdp, plot=False, num_cs=1):
    initial_params = None

    if problem == 1:
        space = [Real(name='alpha', low=0.0, high=0.5),
                 Real(name='alpha2', low=0.0, high=0.5)]
        @use_named_args(space)
        def objective(alpha, alpha2):
            return -classification.classify(graphs, labels, contrast_subgraph_graphs_to_points,
                                            alpha=alpha, alpha2=alpha2, problem=1, solver=solver,
                                            num_cs=num_cs)
        if initial_alpha:
            if initial_alpha2:
                initial_params = [initial_alpha, initial_alpha2]
            else:
                initial_params = [initial_alpha, initial_alpha]
        result = forest_minimize(objective, space, n_calls=20, n_initial_points=5, x0=initial_params, random_state=23)
        if(plot):
            plot_convergence(result)
            plt.savefig("plots/convergence.png")
        return result.x

    elif problem == 2:
        space = [Real(name='alpha', low=0.0, high=0.5)]
        @use_named_args(space)
        def objective(alpha):
            return -classification.classify(graphs, labels, contrast_subgraph_graphs_to_points,
                                            alpha=alpha, problem=2, solver=solver, num_cs=num_cs)
        
        initial_params = [initial_alpha] if initial_alpha else None
        result = forest_minimize(objective, space, n_calls=20, n_initial_points=5, x0=initial_params, random_state=23)
        if(plot):
            plot_convergence(result)
            plt.savefig("plots/convergence.png")
        return result.x[0]
    else:
        print("Cannot tune alpha - Incorrect value for problem formulation")
        return
    

def contrast_subgraph_graphs_to_points(train_graphs, train_labels, test_graphs, alpha=0.05, alpha2=None,
                                        problem=1, solver=dense_subgraph.sdp, important_nodes=[], num_cs=1):
    # Create and Write Summary Graphs
    summary_A = utils.summary_graph(train_graphs[np.where(train_labels == utils.A_LABEL)])
    summary_B = utils.summary_graph(train_graphs[np.where(train_labels == utils.B_LABEL)])

    nodes = np.arange(summary_A.shape[0])
    train_points = np.zeros((train_graphs.shape[0], 2))
    test_points = np.zeros((test_graphs.shape[0], 2))
    # Get the difference network between the edge weights in group A and B
    if problem == 1:
        node_mask_a_b = np.array([True]*nodes.shape[0])
        node_mask_b_a = np.array([True]*nodes.shape[0])
        cs_a_b = np.array([], dtype=int)
        cs_b_a = np.array([], dtype=int)
        diff_a_b = summary_A - summary_B
        diff_b_a = summary_B - summary_A

        for i in range(num_cs):
            masked_diff_a_b = utils.induce_subgraph(diff_a_b, nodes[node_mask_a_b])
            masked_diff_b_a = utils.induce_subgraph(diff_b_a, nodes[node_mask_b_a])
            cs_a_b = nodes[node_mask_a_b][solver(masked_diff_a_b, alpha)]
            cs_b_a = nodes[node_mask_b_a][solver(masked_diff_b_a, alpha2 if alpha2 else alpha)]
            # Do not consider the previously found contrast subgraph nodes for future contrast subgraphs
            node_mask_a_b[cs_a_b] = False
            node_mask_b_a[cs_b_a] = False

            train_points += cs_p1_graphs_to_points(train_graphs, cs_a_b, cs_b_a)
            test_points += cs_p1_graphs_to_points(test_graphs, cs_a_b, cs_b_a)
            if len(nodes[node_mask_a_b]) == 0:
                print("Every node in the graph is included by a contrast subgraph(A->B)!\n\
                    Stopped at Contrast Subgraph {}.".format(i+1))
                break
            if len(nodes[node_mask_b_a]) == 0:
                print("Every node in the graph is included by a contrast subgraph (B->A)!\n\
                    Stopped at Contrast Subgraph {}.".format(i+1))
                break

        return train_points, test_points
    else:
        node_mask = np.array([True]*nodes.shape[0])
        cs = np.array([], dtype=int)
        diff = abs(summary_A - summary_B)
        
        for i in range(num_cs):
            masked_diff = utils.induce_subgraph(diff, nodes[node_mask])
            cs = nodes[node_mask][solver(masked_diff, alpha)]
            node_mask[cs] = False

            train_points += cs_p2_graphs_to_points(train_graphs, cs, summary_A, summary_B)
            test_points += cs_p2_graphs_to_points(test_graphs, cs, summary_A, summary_B)
            if len(nodes[node_mask]) == 0:
                print("Every node in the graph is included by a contrast subgraph!\n\
                    Stopped at Contrast Subgraph {}.".format(i+1))
                break
        
        return train_points, test_points

def main():
    parser = argparse.ArgumentParser(description='Graph Classification via Contrast Subgraphs')
    parser.add_argument('A_dir', help='Filepath to class A directory containing brain network files.', type=str)
    parser.add_argument('B_dir', help='Filepath to class B directory containing brain network files.', type=str)
    parser.add_argument('-a','--alpha', help='Penalty value for contrast subgraph size (varies from 0 to 1).\
                        If not provided, --tune-alpha is set to true.', type=float, metavar='a')
    parser.add_argument('-a2','--alpha2', help='A secondary alpha value to use for the contrast subgraph from B to A \
            (only applies if problem formulation is 1). Note that the original alpha is used for both contrast subgraphs \
            if this is not provided.', type=float, metavar='a')
    parser.add_argument('-t', '--tune-alpha', help='Whether or not to tune the alpha hyperparameter(s) before running the cross-validation (increases runtime).\
                        Note that alpha is automatically tuned if no alpha value is provided.', default=False, action="store_true")
    parser.add_argument('-p', '--problem', help='Problem Formulation (default: 1)', type=int, default = 1, choices={1,2})
    parser.add_argument('-k','--num-folds', help='Number of times to fold data in k-fold cross validation (default: 5).', type=int, default = 5)
    parser.add_argument('-s','--solver', help='Solver to use for finding a contrast subgraph (default: sdp).', type=str, default = "sdp", choices={"sdp","qp"})
    parser.add_argument('-loo', '--leave-one-out', help='If present, perform leave-one-out cross validation (can be computationally expensive). This will cause num-folds to be ignored.', default=False, action="store_true")
    parser.add_argument('--plot', help='If present, plots will be generated in the ./plots/ directory.', default=False, action="store_true")
    parser.add_argument('-pre','--plot-prefix', help='A string to prepend to plot names.', type=str, default="")
    parser.add_argument('-cs','--num-contrast-subgraphs', help='Number of non-overlapping contrast subgraphs to use (default: 1).\
        When a cs is found, its nodes are removed from the difference network and the next cs is found.\
        At the end, the contrast subgraphs are concatenated and used together.', type=int, default = 1)

    args = parser.parse_args()
    alpha = args.alpha
    alpha2 = args.alpha2

    if not alpha:
        args.tune_alpha = True
    elif(alpha < 0 or alpha > 1):
        raise ValueError("alpha should be between 0 and 1 inclusive.")
    if(args.problem == 1 and alpha2 and (alpha2 < 0 or alpha2 > 1)):
        raise ValueError("secondary alpha should be between 0 and 1 inclusive.")
    
    if args.solver == "sdp":
        solver = dense_subgraph.sdp
    elif args.solver == "qp":
        solver = dense_subgraph.qp
    
    print("Performing Contrast Subgraph Classification on Brain Networks\n")

    # Read brain graph files into numpy arrays
    graphs_A = utils.get_graphs_from_files(args.A_dir)
    graphs_B = utils.get_graphs_from_files(args.B_dir)

    graphs, labels = utils.get_AB_labels(graphs_A, graphs_B)

    num_folds = args.num_folds
    if args.leave_one_out:
        num_folds = len(labels)

    if args.tune_alpha:
        print("Tuning alpha value(s)...")
        if args.problem == 1:
            alpha, alpha2 = tune_alpha_accuracy( graphs, labels, alpha, alpha2,
                                        problem=args.problem, solver=solver,
                                        plot=args.plot, num_cs=args.num_contrast_subgraphs)
        if args.problem == 2:
            alpha = tune_alpha_accuracy( graphs, labels, alpha,
                                problem=args.problem, solver=solver,
                                plot=args.plot, num_cs=args.num_contrast_subgraphs)

    # Reporting
    print("\nProblem Formulation {} with".format(args.problem), end=" ")
    if args.problem == 1:
        print("A->B alpha = {}, B->A alpha = {}".format(alpha, alpha2 if alpha2 else alpha))
    elif args.problem == 2:
        print("alpha = {}".format(alpha))

    print("Solver: ", args.solver.upper())
    print("Number of Contrast Subgraphs: {}".format(args.num_contrast_subgraphs))
    classification.classify(graphs, labels, contrast_subgraph_graphs_to_points,
                            num_folds, args.leave_one_out, args.plot, args.plot_prefix, random_state=23,
                            alpha=alpha, alpha2=alpha2, problem=args.problem, solver=solver,
                            num_cs=args.num_contrast_subgraphs)
    

if __name__ == "__main__":
    main()