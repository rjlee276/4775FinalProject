import numpy as np
from sklearn.preprocessing import StandardScaler


class CustomGaussianHMM:
    def __init__(self, n_components, n_features, n_iter=100, tol=1e-4, random_state=None):
        # Initialize the HMM parameters
        self.n_components = n_components  # Number of hidden states
        self.n_features = n_features      # Dimension of observation vectors
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        # Initialize parameters
        self.pi = None     # Initial state probabilities
        self.A = None      # Transition probabilities
        self.means = None  # Means of Gaussian emissions
        self.covars = None  # Covariances of Gaussian emissions (diagonal)

    def _initialize_parameters(self, observations_list):
        np.random.seed(self.random_state)
        # Initialize pi uniformly
        self.pi = np.full(self.n_components, 1.0 / self.n_components)
        # Initialize A randomly
        self.A = np.random.rand(self.n_components, self.n_components)
        self.A /= self.A.sum(axis=1, keepdims=True)
        # Initialize means and covariances
        all_obs = np.vstack(observations_list)
        labels = np.random.randint(self.n_components, size=all_obs.shape[0])
        self.means = np.zeros((self.n_components, self.n_features))
        self.covars = np.zeros((self.n_components, self.n_features))
        for i in range(self.n_components):
            obs_i = all_obs[labels == i]
            if len(obs_i) == 0:
                self.means[i] = np.random.randn(self.n_features)
                self.covars[i] = np.ones(self.n_features)
            else:
                self.means[i] = obs_i.mean(axis=0)
                self.covars[i] = obs_i.var(axis=0) + 1e-2

    def _gaussian_pdf(self, x, mean, covar):
        # Compute the probability density of x given mean and covar (diagonal covariance)
        coef = 1.0 / np.sqrt((2 * np.pi) ** self.n_features * np.prod(covar))
        exponent = -0.5 * np.sum(((x - mean) ** 2) / covar)
        return coef * np.exp(exponent)

    def _compute_likelihoods(self, observations):
        T = observations.shape[0]
        B_map = np.zeros((T, self.n_components))
        for t in range(T):
            x_t = observations[t]
            for i in range(self.n_components):
                B_map[t, i] = self._gaussian_pdf(
                    x_t, self.means[i], self.covars[i])
                B_map[t, i] = max(B_map[t, i], 1e-300)  # Prevent underflow
        return B_map

    def _forward(self, observations, B_map):
        T = observations.shape[0]
        alpha = np.zeros((T, self.n_components))
        c = np.zeros(T)
        # Initialization
        alpha[0] = self.pi * B_map[0]
        c[0] = alpha[0].sum()
        alpha[0] /= c[0]
        # Induction
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * B_map[t]
            c[t] = alpha[t].sum()
            alpha[t] /= c[t]
        log_likelihood = np.sum(np.log(c))
        return alpha, c, log_likelihood

    def _backward(self, observations, B_map, c):
        T = observations.shape[0]
        beta = np.zeros((T, self.n_components))
        # Initialization
        beta[T - 1] = 1.0 / c[T - 1]
        # Induction
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (B_map[t + 1] * beta[t + 1])) / c[t]
        return beta

    def _baum_welch(self, observations_list):
        prev_log_likelihood = None
        for n_iter in range(self.n_iter):
            gamma_sum = np.zeros(self.n_components)
            xi_sum = np.zeros((self.n_components, self.n_components))
            pi_new = np.zeros(self.n_components)
            means_new = np.zeros((self.n_components, self.n_features))
            covars_new = np.zeros((self.n_components, self.n_features))
            total_log_likelihood = 0
            for observations in observations_list:
                T = observations.shape[0]
                B_map = self._compute_likelihoods(observations)
                alpha, c, log_likelihood = self._forward(observations, B_map)
                beta = self._backward(observations, B_map, c)
                total_log_likelihood += log_likelihood
                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True)
                xi = np.zeros((T - 1, self.n_components, self.n_components))
                for t in range(T - 1):
                    denom = (alpha[t][:, np.newaxis] * self.A *
                             B_map[t + 1] * beta[t + 1]).sum()
                    xi[t] = (alpha[t][:, np.newaxis] * self.A *
                             B_map[t + 1] * beta[t + 1]) / denom
                gamma_sum += gamma.sum(axis=0)
                xi_sum += xi.sum(axis=0)
                pi_new += gamma[0]
                # Update means and covars
                for i in range(self.n_components):
                    gamma_i = gamma[:, i]
                    means_new[i] += gamma_i @ observations
                    covars_new[i] += gamma_i @ (observations ** 2)
            # Normalize
            self.pi = pi_new / len(observations_list)
            self.A = xi_sum / xi_sum.sum(axis=1, keepdims=True)
            for i in range(self.n_components):
                denom = gamma_sum[i]
                self.means[i] = means_new[i] / denom
                self.covars[i] = covars_new[i] / denom - self.means[i] ** 2
                self.covars[i] += 1e-2  # Prevent zero variance
            # Check convergence
            if prev_log_likelihood is not None:
                if abs(total_log_likelihood - prev_log_likelihood) < self.tol:
                    print(f"Converged at iteration {n_iter}")
                    break
            prev_log_likelihood = total_log_likelihood
        else:
            print(f"Reached maximum iterations ({self.n_iter})")

    def fit(self, observations_list):
        self._initialize_parameters(observations_list)
        self._baum_welch(observations_list)

    def score(self, observations):
        B_map = self._compute_likelihoods(observations)
        _, _, log_likelihood = self._forward(observations, B_map)
        return log_likelihood


def hmm_clustering(motif_vectors, target_length, n_features=4, n_components=3, K=2, max_iter=10, tol=1e-6, random_state=42):
    """
    Custom implementation of HMM-based clustering for motif sequences.
    """
    num_motifs = len(motif_vectors)

    # **Adjust K to be no greater than num_motifs**
    K = min(K, num_motifs)

    # Reshape motif_vectors into sequences of shape (target_length, n_features)
    motif_sequences = [vector.reshape(
        (target_length, n_features)) for vector in motif_vectors]

    # Stack all sequences for standardization
    all_data = np.vstack(motif_sequences)

    # Standardize across all sequences
    scaler_seq = StandardScaler()
    all_data_scaled = scaler_seq.fit_transform(all_data)

    # Split back into individual sequences
    motif_sequences_scaled = []
    start = 0
    for _ in motif_sequences:
        end = start + target_length
        motif_sequences_scaled.append(all_data_scaled[start:end])
        start = end

    # Initialize HMMs
    models = []
    for i in range(K):
        model = CustomGaussianHMM(
            n_components=n_components,
            n_features=n_features,
            n_iter=100,
            random_state=random_state + i,  # Different seed for each HMM
            tol=tol
        )
        models.append(model)

    # **Randomly assign motifs to clusters initially, ensuring each cluster has at least one data point**
    np.random.seed(random_state)
    cluster_assignments = np.zeros(num_motifs, dtype=int)
    cluster_assignments[:K] = np.arange(K)
    if num_motifs > K:
        cluster_assignments[K:] = np.random.randint(0, K, size=num_motifs - K)
    np.random.shuffle(cluster_assignments)

    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}")

        # Re-estimate HMM parameters for each cluster
        for k in range(K):
            indices = [i for i, c in enumerate(cluster_assignments) if c == k]
            # Get the sequences for this cluster
            sequences_cluster = [motif_sequences_scaled[i] for i in indices]
            # Fit the HMM to the data
            models[k].fit(sequences_cluster)

        # Re-assign motifs to clusters
        new_assignments = np.zeros(num_motifs, dtype=int)
        for i in range(num_motifs):
            log_likelihoods = np.array(
                [models[k].score(motif_sequences_scaled[i]) for k in range(K)])
            new_assignments[i] = np.argmax(log_likelihoods)

        # Check for convergence
        if np.array_equal(cluster_assignments, new_assignments):
            print("Converged")
            break

        cluster_assignments = new_assignments

    # Compute log-likelihoods of each sequence under each HMM
    log_likelihood_matrix = np.zeros((num_motifs, K))
    for i in range(num_motifs):
        for k in range(K):
            log_likelihood = models[k].score(motif_sequences_scaled[i])
            log_likelihood_matrix[i, k] = log_likelihood

    # Standardize the log-likelihoods across all sequences
    scaler_ll = StandardScaler()
    log_likelihood_matrix_scaled = scaler_ll.fit_transform(
        log_likelihood_matrix)

    return cluster_assignments, models, log_likelihood_matrix_scaled
