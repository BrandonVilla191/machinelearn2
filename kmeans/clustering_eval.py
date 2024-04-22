from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import numpy as np

def silhouette_coefficient(X, labels):
    return silhouette_score(X, labels) 

def beta_cv(X, labels, n_clusters):
    intra_cluster_dist = []
    nearest_cluster_dist = []

    pairwise_dist = squareform(pdist(X))
    unique_labels = np.unique(labels)

    for c in unique_labels:
        cluster_points_indices = np.where(labels == c)[0]
        cluster_points = X[cluster_points_indices]

        if len(cluster_points) > 1:
            intra_distances = pairwise_dist[cluster_points_indices, :][:, cluster_points_indices]
            avg_intra_dist = np.sum(intra_distances) / (2 * len(cluster_points))
            intra_cluster_dist.append(avg_intra_dist)

        mask = labels != c
        if np.any(mask):  
            nearest_distances = np.min(pairwise_dist[cluster_points_indices, :][:, mask], axis=1)
            avg_nearest_dist = np.mean(nearest_distances)
            nearest_cluster_dist.append(avg_nearest_dist)

    if n_clusters > 1:  
        beta_cv = np.sum(intra_cluster_dist) / np.sum(nearest_cluster_dist)
    else:
        beta_cv = float('inf')  
    return beta_cv