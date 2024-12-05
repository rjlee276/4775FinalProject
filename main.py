import numpy as np
from tqdm import tqdm
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO, AlignIO, Phylo
from Bio.Align.Applications import MafftCommandline
import matplotlib.pyplot as plt
import subprocess
import requests
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Import clustering methods
from clustering_algorithms.kmeds import k_medoids
from clustering_algorithms.hmm import hmm_clustering
from clustering_algorithms.fuzzyc import init_fuzzy_c_means


# Import Preprocessing Utils
from preprocessing.utils import fetch_pfms, pfm_to_pwm, pfm_to_probabilities, pad_or_trim_pwm, pwm_to_vector, pwm_to_consensus, pwm_similarity, create_unique_id, compute_total_entropy, is_valid_pwm

# ------------------ Parameters ------------------
# Define species list
species_list = [
    {"name": "Homo sapiens", "tax_id": "9606"},
    {"name": "Mus musculus", "tax_id": "10090"},
    {"name": "Rattus norvegicus", "tax_id": "10116"},
    {"name": "Gallus gallus", "tax_id": "9031"},
    {"name": "Xenopus laevis", "tax_id": "8355"},
    {"name": "Drosophila melanogaster", "tax_id": "7227"}
]

tf_classes = [
    "basic helix-loop-helix factors",
    "c2h2 zinc finger factors",
    "nuclear receptors with c4 zinc fingers"
]

tf_class_abbr = {
    "basic helix-loop-helix factors": "bHLH",
    "c2h2 zinc finger factors": "C2H2",
    "nuclear receptors with c4 zinc fingers": "ZC4"
}

# Define clustering methods
clustering_methods = {
    'kmedoids': 'custom_kmedoids',
    'hmm': 'custom_hmm',
    'fuzzy_c_means': 'custom_fuzzy_c_means'
}

# Set the maximum number of PFMs to process per TF class
max_pfms_per_class = 50

# ------------------ Main processing loop ------------------
for tf_class in tf_classes:
    tf_abbr = tf_class_abbr.get(tf_class, tf_class)
    motif_data = []  # List to hold motif information
    id_mapping = {}  # Initialize the ID mapping dictionary

    print(f"\nProcessing TF class: {tf_class} ({tf_abbr})")

    # Fetch and process PFMs for each species in the current TF class
    for species in species_list:
        pfms = fetch_pfms(species["tax_id"], tf_class,
                          max_matrices=max_pfms_per_class)
        for matrix in tqdm(pfms, desc=f"Processing {tf_class} for {species['name']}"):
            matrix_id = matrix['matrix_id']
            # Fetch the PFM data for this matrix
            matrix_params = {"format": "json"}
            matrix_response = requests.get(
                f"https://jaspar.genereg.net/api/v1/matrix/{matrix_id}/", params=matrix_params)
            if matrix_response.status_code == 200:
                matrix_data = matrix_response.json()
                pfm_dict = matrix_data["pfm"]
                try:
                    pfm = np.array([
                        pfm_dict['A'],
                        pfm_dict['C'],
                        pfm_dict['G'],
                        pfm_dict['T']
                    ]).T
                except KeyError as e:
                    print(f"KeyError: {e} in matrix {matrix_id}")
                    continue
                # Compute probabilities
                probabilities = pfm_to_probabilities(pfm)
                # Compute PWM
                pwm = pfm_to_pwm(pfm)
                # Validate PWM and probabilities
                if is_valid_pwm(pwm) and is_valid_pwm(probabilities):
                    # Collect PWMs and probabilities
                    motif_data.append({
                        'pwm': pwm,
                        'probabilities': probabilities,
                        'species': species['name'],
                        'species_tax_id': species['tax_id'],
                        'matrix_id': matrix_id
                    })
                else:
                    continue
            else:
                print(f"Failed to fetch data for matrix {matrix_id}")

    if len(motif_data) < 2:
        continue

    # Determine the maximum length of all PWMs
    max_length = max(item['pwm'].shape[0] for item in motif_data)
    target_length = max_length

    # Standardize PWMs and probabilities to the target length
    for item in motif_data:
        standardized_pwm = pad_or_trim_pwm(item['pwm'], target_length)
        standardized_probabilities = pad_or_trim_pwm(
            item['probabilities'], target_length)
        item['pwm'] = standardized_pwm
        item['probabilities'] = standardized_probabilities
        # Generate consensus sequence
        consensus_seq = pwm_to_consensus(standardized_pwm)
        unique_id = create_unique_id(
            {'name': item['species']}, tf_class, item['matrix_id']
        )
        descriptive_label = f"{item['species']} ({tf_abbr})"
        # Store the mapping
        id_mapping[unique_id] = {
            'label': descriptive_label,
            'species': item['species']
        }
        record = SeqRecord(Seq(consensus_seq),
                           id=unique_id,
                           description=descriptive_label)
        item['sequence_record'] = record
        item['unique_id'] = unique_id
        item['descriptive_label'] = descriptive_label

    # Extract vectors
    motif_vectors = np.array([pwm_to_vector(item['pwm'])
                              for item in motif_data])

    # Compute pairwise similarity matrix
    num_motifs = len(motif_data)
    similarity_matrix = np.zeros((num_motifs, num_motifs))
    for i in range(num_motifs):
        for j in range(i, num_motifs):
            sim = pwm_similarity(motif_data[i]['pwm'], motif_data[j]['pwm'])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim  # Symmetric matrix

    # Apply filtering to remove redundant motifs
    similarity_threshold = 0.95
    filtered_indices = []
    for i in range(num_motifs):
        if all(similarity_matrix[i, j] < similarity_threshold for j in filtered_indices):
            filtered_indices.append(i)

    # Filter motif_data and motif_vectors
    filtered_motif_data = [motif_data[i] for i in filtered_indices]
    filtered_motif_vectors = motif_vectors[filtered_indices]

    # Update num_motifs
    num_motifs = len(filtered_motif_data)

    if num_motifs < 2:
        print("Not enough motifs after filtering. Skipping this TF class.")
        continue

    # Recompute similarity_matrix and distance_matrix for filtered motifs
    similarity_matrix = np.zeros((num_motifs, num_motifs))
    for i in range(num_motifs):
        for j in range(i, num_motifs):
            sim = pwm_similarity(
                filtered_motif_data[i]['pwm'], filtered_motif_data[j]['pwm'])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim  # Symmetric matrix

    # Compute total entropy for each motif and add as a feature
    for item in filtered_motif_data:
        entropy = compute_total_entropy(item['probabilities'])
        item['entropy'] = entropy

    # Concatenate PWM vectors and entropy for clustering
    clustering_features = np.hstack([
        filtered_motif_vectors,
        np.array([item['entropy']
                  for item in filtered_motif_data]).reshape(-1, 1)
    ])

    # Check for NaNs or infinite values in clustering_features
    if np.isnan(clustering_features).any() or np.isinf(clustering_features).any():
        print("NaNs or infinite values detected in clustering features. Cleaning data...")
        valid_indices = ~np.isnan(clustering_features).any(
            axis=1) & ~np.isinf(clustering_features).any(axis=1)
        clustering_features = clustering_features[valid_indices]
        filtered_motif_data = [filtered_motif_data[i] for i in range(
            len(filtered_motif_data)) if valid_indices[i]]

    if clustering_features.shape[0] < 2:
        print("Not enough valid motifs after cleaning. Skipping this TF class.")
        continue

    # Scale the features
    scaler = StandardScaler()
    clustering_features_scaled = scaler.fit_transform(clustering_features)

    # Determine the number of samples
    n_samples = clustering_features_scaled.shape[0]
    print(f"Number of samples after filtering: {n_samples}")

    # Set perplexity for t-SNE
    if n_samples <= 5:
        # Ensure perplexity is less than n_samples
        perplexity = max(1, n_samples - 2)
    else:
        perplexity = min(30, (n_samples - 1) // 3)
    print(f"Using perplexity: {perplexity}")

    # Apply t-SNE for visualization
    reduced_features = TSNE(n_components=2, perplexity=perplexity,
                            random_state=42).fit_transform(clustering_features_scaled)

    for method_name, clustering_model in clustering_methods.items():
        print(f"\nClustering motifs using {method_name} clustering...")

        # Perform clustering
        if method_name == 'kmedoids':
            # Ensure k does not exceed the number of samples
            k = min(5, n_samples)
            labels, medoid_indices = k_medoids(
                clustering_features_scaled, k=k, random_state=42)
        elif method_name == 'hmm':
            # Parameters for HMM clustering
            target_length = filtered_motif_data[0]['pwm'].shape[0]
            n_features = 4  # Assuming A, C, G, T
            n_components = 3
            K = min(5, num_motifs)  # Number of clusters
            cluster_assignments, models, log_likelihood_matrix_scaled = hmm_clustering(
                filtered_motif_vectors,
                target_length=target_length,
                n_features=n_features,
                n_components=n_components,
                K=K,
                max_iter=100,
                tol=1e-6,
                random_state=42
            )
            labels = cluster_assignments
        elif method_name == 'fuzzy_c_means':
            n_clusters = max(3, min(5, n_samples))
            m = 1.25
            max_iter = 200
            error = 1e-6
            cluster_centers, membership_matrix, cluster_assignments = init_fuzzy_c_means(
                clustering_features_scaled, n_clusters, m=m, max_iter=max_iter, error=error)
            labels = cluster_assignments
        else:
            labels = clustering_model.fit_predict(clustering_features_scaled)

        # Assign cluster labels to motif_data
        for i, item in enumerate(filtered_motif_data):
            item[f'{method_name}_label'] = labels[i]

        # Calculate silhouette score
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1 and len(unique_labels) < n_samples:
            sil_score = silhouette_score(clustering_features_scaled, labels)
            print(f"Silhouette Score for {method_name}: {sil_score}")
        else:
            print(f"Silhouette Score not applicable for {method_name}")

        # Visualize clusters using t-SNE reduced features
        plt.figure(figsize=(8, 6))
        plt.scatter(
            reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='tab10')
        plt.title(f"{method_name.capitalize()} Clustering of {tf_abbr} Motifs")
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.colorbar()
        plt.show()

        # Select representative motifs from each cluster
        cluster_representatives = []
        for cluster_label in np.unique(labels):
            cluster_indices = np.where(labels == cluster_label)[0]
            if len(cluster_indices) == 0:
                continue
            if method_name == 'kmedoids':
                # For custom k-medoids, medoid indices are known
                medoid_index = medoid_indices[cluster_label]
            elif method_name == 'hmm':
                # Select the motif with the highest log-likelihood in the cluster
                cluster_ll = log_likelihood_matrix_scaled[cluster_indices, cluster_label]
                best_index = cluster_indices[np.argmax(cluster_ll)]
                medoid_index = best_index
            elif method_name == 'fuzzy_c_means':
                # Select the motif with the highest membership in the cluster
                memberships = membership_matrix[cluster_label, cluster_indices]
                best_index = cluster_indices[np.argmax(memberships)]
                medoid_index = best_index
            else:
                # For other methods, select the motif closest to the cluster centroid
                cluster_features = clustering_features_scaled[cluster_indices]
                centroid = np.mean(cluster_features, axis=0)
                distances = np.linalg.norm(
                    cluster_features - centroid, axis=1)
                closest_index = np.argmin(distances)
                medoid_index = cluster_indices[closest_index]
            representative_motif = filtered_motif_data[medoid_index]
            cluster_representatives.append(representative_motif)

        # Collect sequences of representatives
        rep_sequences = [item['sequence_record']
                         for item in cluster_representatives]

        if len(rep_sequences) < 3:
            continue

        # Save representative sequences to FASTA file
        rep_fasta_filename = f"{method_name}_representatives_{tf_abbr}.fasta"
        SeqIO.write(rep_sequences, rep_fasta_filename, "fasta")

        # Perform MSA using MAFFT
        aligned_filename = f"{method_name}_aligned_{tf_abbr}.fasta"
        mafft_cline = MafftCommandline(input=rep_fasta_filename)
        stdout, stderr = mafft_cline()
        with open(aligned_filename, "w") as handle:
            handle.write(stdout)

        # Read the alignment
        alignment = AlignIO.read(aligned_filename, "fasta")

        # Construct the phylogenetic tree using Maximum Likelihood with PhyML
        phyml_input = f"{method_name}_phyml_input_{tf_abbr}.phy"
        phylip_id_mapping = {}  # Map PHYLIP IDs to original IDs

        # Assign new IDs
        for idx, record in enumerate(alignment):
            original_id = record.id  # Original long ID
            new_id = f"id{idx:02}"   # Short unique ID for PHYLIP
            # Map new ID to original ID
            phylip_id_mapping[new_id] = original_id
            record.id = new_id       # Update the record ID

        # Write the alignment to PHYLIP format
        AlignIO.write(alignment, phyml_input, "phylip-relaxed")

        # Run PhyML
        phyml_command = f"phyml -i {phyml_input} -d nt -b 100"
        print(f"Running PhyML for {method_name} clustering...")
        subprocess.run(phyml_command, shell=True)

        # Read the resulting tree
        tree_filename = f"{phyml_input}_phyml_tree.txt"
        tree = Phylo.read(tree_filename, "newick")

        # Replace terminal node names with descriptive labels
        for terminal in tree.get_terminals():
            phylip_id = terminal.name
            original_id = phylip_id_mapping.get(phylip_id)
            if original_id:
                info = id_mapping.get(original_id)
                if info:
                    terminal.name = info['label']
                else:
                    print(f"Warning: Original ID {original_id}")
            else:
                print(f"Warning: PHYLIP ID {phylip_id}")

        # Visualize the tree
        fig = plt.figure(figsize=(10, 8))
        axes = fig.add_subplot(1, 1, 1)
        Phylo.draw(tree, axes=axes, do_show=False)
        plt.title(
            f"Phylogenetic Tree ({method_name.capitalize()} Clustering) - {tf_abbr}")
        plt.show()

    # Plot the distribution of pairwise similarities after filtering
    plt.hist(similarity_matrix.flatten(), bins=50)
    plt.title("Distribution of Pairwise Similarities After Filtering")
    plt.xlabel("Similarity")
    plt.ylabel("Frequency")
    plt.show()
