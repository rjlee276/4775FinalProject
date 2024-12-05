import numpy as np


def k_medoids(X, k, max_iter=300, random_state=None):
    """
    Custom implementation of k-medoids clustering.

    Parameters:
    - X: numpy.ndarray, shape (n_samples, n_features)
        The data matrix.
    - k: int
        The number of clusters.
    - max_iter: int, optional (default=300)
        Maximum number of iterations.
    - random_state: int or None, optional
        Random seed for reproducibility.

    Returns:
    - labels: numpy.ndarray, shape (n_samples,)
        Cluster labels for each point.
    - medoid_indices: numpy.ndarray, shape (k,)
        Indices of the medoids in the original dataset.
    """
    m, n = X.shape
    # Randomly initialize medoids
    if random_state is not None:
        np.random.seed(random_state)
    medoid_indices = np.random.choice(m, k, replace=False)
    medoids = X[medoid_indices]
    labels = np.zeros(m, dtype=int)

    for iteration in range(max_iter):
        # Assign each point to the nearest medoid
        distances = np.linalg.norm(X[:, np.newaxis] - medoids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Update medoids
        new_medoid_indices = np.copy(medoid_indices)
        for j in range(k):
            cluster_indices = np.where(labels == j)[0]
            if len(cluster_indices) == 0:
                continue
            # Compute the cost for each point in the cluster
            cluster_points = X[cluster_indices]
            intra_distances = np.sum(
                np.linalg.norm(
                    cluster_points[:, np.newaxis] - cluster_points, axis=2),
                axis=1,
            )
            min_index = cluster_indices[np.argmin(intra_distances)]
            new_medoid_indices[j] = min_index

        # Check for convergence
        if np.array_equal(new_medoid_indices, medoid_indices):
            print(f"Converged after {iteration + 1} iterations.")
            break
        medoid_indices = new_medoid_indices
        medoids = X[medoid_indices]
    else:
        print(f"Reached maximum iterations ({max_iter}).")

    return labels, medoid_indices
