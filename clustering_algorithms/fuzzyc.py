import numpy as np
from sklearn.cluster import KMeans


def initialize_membership_matrix(n_samples, n_clusters, data):
    """Initialize the membership matrix using KMeans."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
    labels = kmeans.labels_
    u = np.zeros((n_clusters, n_samples))
    u[labels, np.arange(n_samples)] = 1
    return u


def compute_cluster_centers(u, data, m):
    um = u ** m
    numerator = um @ data
    denominator = np.sum(um, axis=1, keepdims=True)
    centers = numerator / denominator
    return centers


def update_membership_matrix(data, centers, m):
    n_samples = data.shape[0]
    n_clusters = centers.shape[0]
    distances = np.zeros((n_samples, n_clusters))
    for k in range(n_clusters):
        distances[:, k] = np.linalg.norm(data - centers[k], axis=1) ** 2
    distances = np.fmax(distances, np.finfo(np.float64).eps)
    inv_distances = distances ** (-1 / (m - 1))
    u_new = inv_distances / np.sum(inv_distances, axis=1, keepdims=True)
    return u_new.T


def fuzzy_c_means(data, n_clusters, m=1.25, max_iter=500, error=1e-7):
    n_samples, n_features = data.shape
    u = initialize_membership_matrix(n_samples, n_clusters, data)

    for iteration in range(max_iter):
        u_old = u.copy()
        centers = compute_cluster_centers(u, data, m)
        u = update_membership_matrix(data, centers, m)

        diff = np.linalg.norm(u - u_old)
        if diff < error:
            print(
                f"Fuzzy C-Means converged at iteration {iteration}, diff: {diff}")
            break

    return centers, u


def assign_clusters(membership_matrix):
    return np.argmax(membership_matrix, axis=0)


def init_fuzzy_c_means(motif_vectors, n_clusters, m=1.25, max_iter=500, error=1e-7):
    motif_vectors = np.array(motif_vectors)
    motif_vectors = motif_vectors.reshape(motif_vectors.shape[0], -1)
    cluster_centers, membership_matrix = fuzzy_c_means(
        motif_vectors, n_clusters, m=m, max_iter=max_iter, error=error)
    cluster_assignments = assign_clusters(membership_matrix)
    return cluster_centers, membership_matrix, cluster_assignments
