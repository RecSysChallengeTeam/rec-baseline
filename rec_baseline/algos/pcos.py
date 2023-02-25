import numpy as np
import pandas as pd

from typing import Dict


class PCos:
    """
    Popularity-Aware Cosine Similarity.
    
    Ref: https://dl.acm.org/doi/abs/10.1145/3556702.3556851
    """
    
    def __init__(self, item2vec: Dict):
        self.item2vec = item2vec
        self.cos_sim_mat = None
        self.conf_mat = None
        
    def fit(self, views: pd.Series, purchase: pd.Series):
        # Cosine similarity matrix
        item_vector_matrix = [item_vector for _, item_vector in self.item2vec.items()]
        self.cos_sim_mat = item_vector_matrix @ item_vector_matrix.T
        
        # Confidence matrix
        conf_mat = np.ones_like(self.cos_sim_mat)
        support = np.ones(self.cos_sim_mat.shape[0])
        
        for view, purchase in zip(views, purchase):
            unique_view = np.unique(view)
            for item_i, item_j in zip(unique_view, unique_view[1:]):
                conf_mat[item_i, item_j] += 1
                conf_mat[item_j, item_i] += 1
                conf_mat[item_i, purchase] += 3
                conf_mat[purchase, item_i] += 3
                conf_mat[item_j, purchase] += 3
                conf_mat[purchase, item_j] += 3
                
                support[item_i] += 1
                support[item_j] += 1
            support[purchase] += 3
            
        self.conf_mat = conf_mat / support
        
    def predict(self, views: pd.Series, weights: np.ndarray = [0.01, 0.1, 0.9]) -> np.ndarray:
        """
        # TODO: currently not support batch prediction
        
        Args:
            views: user views
            weights: weights of the three components
            
        Returns:
            ordered item ids
        """
        
        k = len(weights) if weights else None
        
        cos_sim = np.array([
            weight * self.cos_sim[view] for weight, view in zip(weights, views[-k:])
        ]).sum(axis=0)  # (n_items,)
        
        conf = np.array([
            weight * self.conf[view] for weight, view in zip(weights, views[-k:])
        ]).sum(axis=0)  # (n_items,)
        
        score = cos_sim * conf
        
        ordered_item_ids = np.argsort(score)[::-1]
        
        return ordered_item_ids
