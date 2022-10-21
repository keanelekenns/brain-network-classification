import numpy as np
from pipeline import Pipeline
import utils


class DiscriminativeEdgesTransformer():
    # Class var to indicate that pipeline can be given to this class's constructor.
    takes_pipeline = True

    def __init__(self, a_label: str, b_label: str, num_edges: int, pipeline: Pipeline = None) -> None:
        self.a_label = a_label
        self.b_label = b_label

        self.num_edges = num_edges

        if pipeline is not None:
            pipeline.axes_labels = [f"% similarity between important {a_label} edges",
                                    f"% similarity between important {b_label} edges",
                                    f"% similarity of whole graph with {a_label} class"]
            pipeline.a_label = a_label
            pipeline.b_label = b_label
            pipeline.plot_prefix = f"DE-{num_edges}"
        
        self.pipeline = pipeline

    def fit(self, graphs, labels) -> None:
        # Create and Write Summary Graphs
        summary_A = utils.summary_graph(graphs[np.where(labels == self.a_label)])
        summary_B = utils.summary_graph(graphs[np.where(labels == self.b_label)])
            
        # Get the difference network between the edge weights in group A and B
        # Note that (u,v) is the same as (v,u), so we extract the upper triangle of the matrix
        self.diff_net = np.triu(summary_A - summary_B, k=1)

        # Find the num_edges most positive and most negative edge diffs
        partitions = np.argpartition(self.diff_net, (self.num_edges, -self.num_edges), axis=None)
        top_n = np.unravel_index(partitions[-self.num_edges:], self.diff_net.shape)
        bottom_n = np.unravel_index(partitions[:self.num_edges], self.diff_net.shape)

        # Ensure the top edges are all positive and the bottom edges are all negative
        top_edges = self.diff_net[top_n]
        positive = top_edges > 0
        self.positive_indices = (top_n[0][positive], top_n[1][positive])
        self.important_a_edges = self.diff_net[self.positive_indices]

        bottom_edges = self.diff_net[bottom_n]
        negative = bottom_edges < 0
        self.negative_indices = (bottom_n[0][negative], bottom_n[1][negative])
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
        graph[np.where(graph==0)] = -1
        return np.array([100*np.dot(self.important_a_edges, graph[self.positive_indices])/self.a_sum,
                         100*np.dot(self.important_b_edges, graph[self.negative_indices])/self.b_sum,
                         100*np.sum(np.multiply(graph, self.diff_net))/self.full_sum])