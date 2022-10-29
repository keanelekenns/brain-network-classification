import numpy as np


class IidakaTransformer():

    def __init__(self, a_label: str, b_label: str, effect_size_threshold: int) -> None:
        self.a_label = a_label
        self.b_label = b_label

        self.effect_size_threshold = effect_size_threshold

    def fit(self, graphs, labels) -> None:

        z_graphs = np.copy(graphs)
        for i in range(z_graphs.shape[0]):
            np.fill_diagonal(z_graphs[i,:,:], 0)
        
        # Fisher's Z Transform (assumes input graphs are correlation matrices)
        z_graphs = np.arctanh(z_graphs)

        # Note that (u,v) is the same as (v,u), so we extract the upper triangle of the matrices
        a_graphs = np.triu(z_graphs[np.where(labels == self.a_label)], k=1)
        b_graphs = np.triu(z_graphs[np.where(labels == self.b_label)], k=1)
            
        a_mean = np.mean(a_graphs, axis=0)
        b_mean = np.mean(b_graphs, axis=0)
        diff_mean = a_mean - b_mean

        a_std = np.std(a_graphs, axis=0)
        b_std = np.std(b_graphs, axis=0)
        pooled_std = np.sqrt((np.square(a_std) + np.square(b_std))/2)

        self.cohens_d = np.absolute(np.divide(diff_mean, pooled_std, out=np.zeros_like(diff_mean), where=pooled_std!=0))

        self.feature_indices = np.where(self.cohens_d > self.effect_size_threshold)
        
        return self

    
    def transform(self, graphs):
        points = np.array(list(map(lambda graph: graph[self.feature_indices], graphs)))
        return points