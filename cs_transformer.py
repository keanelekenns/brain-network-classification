import numpy as np
from pipeline import Pipeline
import utils


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


class ContrastSubgraphTransformer():
    # Class var to indicate that pipeline can be given to this class's constructor.
    takes_pipeline = True
    
    def __init__(self, a_label=None, b_label=None,
                    alpha=None, alpha2=None,
                    percentile=None, percentile2=None,
                    problem=None, solver=None, num_cs=None,
                    pipeline:Pipeline=None) -> None:
        self.a_label = a_label
        self.b_label = b_label

        self.alpha = alpha
        self.alpha2 = alpha2 or alpha
        self.percentile = percentile
        self.percentile2 = percentile2 or percentile
        
        self.problem = problem
        self.solver = solver
        self.num_cs = num_cs

        self.cs_a_b_list = []
        self.cs_b_a_list = []
        self.cs_list = []

        self.alpha_provided = bool(self.alpha)

        if pipeline is not None:
            if problem == 1:
                pipeline.axes_labels = [f"Number of edges overlapping with CS {a_label}-{b_label}",
                                        f"Number of edges overlapping with CS {b_label}-{a_label}"]
            else: # problem == 2
                pipeline.axes_labels = [r"L1 norm distance from $G^{%s}$"%a_label,
                                        r"L1 norm distance from $G^{%s}$"%b_label]

            pipeline.a_label = a_label
            pipeline.b_label = b_label
            pipeline.plot_prefix = f"CSP{problem}-{solver.__name__.upper()}-N{num_cs}"
        
        self.pipeline = pipeline

    def fit(self, X, y=None):
         # Create and Write Summary Graphs
        self.summary_A = utils.summary_graph(X[np.where(y == self.a_label)])
        self.summary_B = utils.summary_graph(X[np.where(y == self.b_label)])

        if self.problem == 1:
            self.find_cs_p1()
        else: # problem == 2
            self.find_cs_p2()
        
        return self

    def find_cs_p1(self) -> None:
        diff_a_b = self.summary_A - self.summary_B
        diff_b_a = self.summary_B - self.summary_A

        nodes = np.arange(diff_a_b.shape[0])
        node_mask_a_b = np.array([True]*nodes.shape[0])
        node_mask_b_a = np.array([True]*nodes.shape[0])

        for i in range(self.num_cs):
            masked_diff_a_b = utils.induce_subgraph(diff_a_b, nodes[node_mask_a_b])
            masked_diff_b_a = utils.induce_subgraph(diff_b_a, nodes[node_mask_b_a])

            # If no alpha value is provided, find the appropriate alpha value using the given percentile
            if not self.alpha_provided:
                # A -> B
                flat = masked_diff_a_b[np.triu_indices_from(masked_diff_a_b, k=1)]
                self.alpha = np.percentile(flat, self.percentile)
                print("alpha = {} ({}-th percentile)".format(self.alpha, self.percentile), end=", ")

                # B -> A
                flat = masked_diff_b_a[np.triu_indices_from(masked_diff_b_a, k=1)]
                self.alpha2 = np.percentile(flat, self.percentile2)
                print("alpha2 = {} ({}-th percentile)".format(self.alpha2, self.percentile2))

            cs_a_b = nodes[node_mask_a_b][self.solver(masked_diff_a_b, self.alpha)]
            self.cs_a_b_list.append(cs_a_b)
            cs_b_a = nodes[node_mask_b_a][self.solver(masked_diff_b_a, self.alpha2)]
            self.cs_b_a_list.append(cs_b_a)
            # Do not consider the previously found contrast subgraph nodes for future contrast subgraphs
            node_mask_a_b[cs_a_b] = False
            node_mask_b_a[cs_b_a] = False

            if len(nodes[node_mask_a_b]) == 0:
                print("Every node in the graph is included by a contrast subgraph(A->B)!\n\
                    Stopped at Contrast Subgraph {}.".format(i+1))
                break
            if len(nodes[node_mask_b_a]) == 0:
                print("Every node in the graph is included by a contrast subgraph (B->A)!\n\
                    Stopped at Contrast Subgraph {}.".format(i+1))
                break

    def find_cs_p2(self) -> None:
        diff = abs(self.summary_A - self.summary_B)

        nodes = np.arange(diff.shape[0])
        node_mask = np.array([True]*nodes.shape[0])
        
        for i in range(self.num_cs):
            masked_diff = utils.induce_subgraph(diff, nodes[node_mask])

            # If no alpha value is provided, find the appropriate alpha value using the given percentile
            if not self.alpha_provided:
                flat = masked_diff[np.triu_indices_from(masked_diff, k=1)]
                self.alpha = np.percentile(flat, self.percentile)
                print("alpha = {} ({}-th percentile)".format(self.alpha, self.percentile))
            
            cs = nodes[node_mask][self.solver(masked_diff, self.alpha)]
            self.cs_list.append(cs)
            node_mask[cs] = False

            if len(nodes[node_mask]) == 0:
                print("Every node in the graph is included by a contrast subgraph!\n\
                    Stopped at Contrast Subgraph {}.".format(i+1))
                break

    def transform(self, X):
        points = np.zeros((X.shape[0], 2))

        if self.problem == 1:
            num_cs = min(len(self.cs_a_b_list), len(self.cs_b_a_list))
            for i in range(num_cs):
                points += cs_p1_graphs_to_points(graphs=X,
                                                cs_a_b=self.cs_a_b_list[i],
                                                cs_b_a=self.cs_b_a_list[i])
        else: # problem == 2
            for i in range(len(self.cs_list)):
                points += cs_p2_graphs_to_points(graphs=X,
                                                contrast_subgraph=self.cs_list[i],
                                                summary_A=self.summary_A,
                                                summary_B=self.summary_B)
        if self.pipeline:
            self.pipeline.add_points(points=points)
        return points