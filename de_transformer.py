import numpy as np
from pipeline import Pipeline
import utils


class DiscriminativeEdgesTransformer():
    # Class var to indicate that pipeline can be given to this class's constructor.
    takes_pipeline = True

    def __init__(self, a_label: str, b_label: str, num_edges: int, pipeline: Pipeline = None, weighted = False) -> None:
        self.a_label = a_label
        self.b_label = b_label

        self.num_edges = num_edges

        self.weighted = weighted

        if pipeline is not None:
            pipeline.axes_labels = [f"% similarity w/ {a_label} DE",
                                    f"% similarity w/ {b_label} DE",
                                    f"% similarity w/ {a_label} class"]
            pipeline.a_label = a_label
            pipeline.b_label = b_label
            pipeline.plot_prefix = f"DE-{num_edges}"
        
        self.pipeline = pipeline

    def fit(self, graphs, labels) -> None:
        # Create and Write Summary Graphs
        # Note that (u,v) is the same as (v,u), so we extract the upper triangle of the matrices
        self.summary_A = np.triu(utils.summary_graph(graphs[np.where(labels == self.a_label)]), k=1)
        self.summary_B = np.triu(utils.summary_graph(graphs[np.where(labels == self.b_label)]), k=1)
            
        # Get the difference network between the edge weights in group A and B
        self.diff_net = self.summary_A - self.summary_B

        # Find the num_edges most positive and most negative edge diffs
        partitions = np.argpartition(self.diff_net, (self.num_edges, -self.num_edges), axis=None)
        top_n = np.unravel_index(partitions[-self.num_edges:], self.diff_net.shape)
        bottom_n = np.unravel_index(partitions[:self.num_edges], self.diff_net.shape)

        # Ensure the top edges are all positive and the bottom edges are all negative
        top_edges = self.diff_net[top_n]
        positive = top_edges > 0
        self.positive_indices = (top_n[0][positive], top_n[1][positive])

        if len(self.positive_indices[0]) < self.num_edges:
            print(f"WARNING: only found {len(self.positive_indices)} positive DEs (looking for {self.num_edges}).")

        bottom_edges = self.diff_net[bottom_n]
        negative = bottom_edges < 0
        self.negative_indices = (bottom_n[0][negative], bottom_n[1][negative])

        if len(self.negative_indices[0]) < self.num_edges:
            print(f"WARNING: only found {len(self.negative_indices)} negative DEs (looking for {self.num_edges}).")

        if self.weighted:
            self.scaled_a_summary = np.multiply(self.summary_A, self.diff_net)
            self.scaled_b_summary = np.multiply(self.summary_B, self.diff_net)
        else:
            self.important_a_edges = self.diff_net[self.positive_indices]
            self.important_b_edges = self.diff_net[self.negative_indices]
            self.a_sum = np.sum(self.important_a_edges)
            self.b_sum = np.sum(self.important_b_edges)
            self.full_sum = np.sum(np.abs(self.diff_net))

        return self

    
    def transform(self, graphs):
        points = np.array(list(map(self.graph_to_point, graphs)))
        if self.pipeline:
            self.pipeline.add_points(points=points)
        return points

    def graph_to_point(self, graph):
        if self.weighted:

            scaled_graph = np.multiply(graph, self.diff_net)
            diff_A = self.scaled_a_summary - scaled_graph
            diff_B = self.scaled_b_summary - scaled_graph
            diff_A_a_edges = diff_A[self.positive_indices]
            diff_A_b_edges = diff_A[self.negative_indices]
            diff_B_a_edges = diff_B[self.positive_indices]
            diff_B_b_edges = diff_B[self.negative_indices]

            A_sim_a_edges = (np.linalg.norm(diff_B_a_edges) - np.linalg.norm(diff_A_a_edges))/(np.linalg.norm(diff_B_a_edges) + np.linalg.norm(diff_A_a_edges))
            B_sim_b_edges = (np.linalg.norm(diff_A_b_edges) - np.linalg.norm(diff_B_b_edges))/(np.linalg.norm(diff_A_b_edges) + np.linalg.norm(diff_B_b_edges))
            A_sim_all = (np.linalg.norm(diff_B) - np.linalg.norm(diff_A))/(np.linalg.norm(diff_B) + np.linalg.norm(diff_A))

            return np.array([100*A_sim_a_edges,
                             100*B_sim_b_edges,
                             100*A_sim_all])
        else:
            graph[np.where(graph==0)] = -1
            return np.array([100*np.dot(self.important_a_edges, graph[self.positive_indices])/self.a_sum,
                         100*np.dot(self.important_b_edges, graph[self.negative_indices])/self.b_sum,
                         100*np.sum(np.multiply(graph, self.diff_net))/self.full_sum])