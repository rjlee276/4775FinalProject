import requests
import numpy as np


def fetch_pfms(tax_id, tf_class, max_matrices=None):
    url = "https://jaspar.genereg.net/api/v1/matrix/"
    params = {
        "tax_id": tax_id,
        "tf_class": tf_class,
        "collection": "CORE",
        "format": "json",
    }
    matrices = []
    while url:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            matrices.extend(results)
            if max_matrices and len(matrices) >= max_matrices:
                matrices = matrices[:max_matrices]
                break
            url = data.get('next')
            params = None
        else:
            break
    return matrices


def pfm_to_pwm(pfm, background_freq=0.25, pseudocount=1e-6):
    """
    Converts a Position Frequency Matrix (PFM) to a Position Weight Matrix (PWM).
    Adds a small pseudocount to avoid log(0).
    """
    pwm = []
    for position in pfm:
        total = np.sum(position) + pseudocount * 4
        pwm_row = []
        for count in position:
            prob = (count + pseudocount) / total
            prob = max(prob, 1e-6)  # Ensure prob is at least 1e-6
            ratio = prob / background_freq
            pwm_value = np.log2(ratio)
            pwm_row.append(pwm_value)
        pwm.append(pwm_row)
    pwm = np.array(pwm)
    # Check for NaNs or infinite values
    if np.isnan(pwm).any() or np.isinf(pwm).any():
        print("Warning: NaNs or infinite values detected in PWM.")
    return pwm


def pfm_to_probabilities(pfm, pseudocount=1e-6):
    """
    Converts a Position Frequency Matrix (PFM) to probabilities.
    """
    probabilities = []
    for position in pfm:
        total = np.sum(position) + pseudocount * 4
        prob_row = [(count + pseudocount) / total for count in position]
        probabilities.append(prob_row)
    probabilities = np.array(probabilities)
    # Clip probabilities to avoid zeros or ones
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return probabilities


def pad_or_trim_pwm(pwm, target_length):
    """Pads or trims a PWM to a target length."""
    current_length = pwm.shape[0]
    if current_length > target_length:
        # Trim the PWM to the target length
        return pwm[:target_length, :]
    elif current_length < target_length:
        # Pad the PWM with zeros to the target length
        padding = np.zeros((target_length - current_length, pwm.shape[1]))
        return np.vstack((pwm, padding))
    else:
        return pwm


def pwm_to_vector(pwm):
    """Flattens a PWM into a single vector."""
    return pwm.flatten()


def pwm_to_consensus(pwm):
    """Converts a PWM into a consensus sequence."""
    consensus_seq = ''
    nucleotides = ['A', 'C', 'G', 'T']
    for position in pwm:
        max_index = np.argmax(position)
        consensus_nucleotide = nucleotides[max_index]
        consensus_seq += consensus_nucleotide
    return consensus_seq


def pwm_similarity(pwm1, pwm2):
    """Calculates similarity between two PWMs using Pearson correlation."""
    vec1 = pwm_to_vector(pwm1)
    vec2 = pwm_to_vector(pwm2)
    # Ensure vectors are of the same length
    if len(vec1) != len(vec2):
        min_length = min(len(vec1), len(vec2))
        vec1 = vec1[:min_length]
        vec2 = vec2[:min_length]
    # Handle zero variance case
    if np.std(vec1) == 0 or np.std(vec2) == 0:
        return 0
    return np.corrcoef(vec1, vec2)[0, 1]


def create_unique_id(species, tf_class, matrix_id):
    species_name = species['name'].replace(' ', '_')
    tf_class_clean = tf_class.replace(' ', '_')
    return f"{species_name}_{tf_class_clean}_{matrix_id}"


def compute_total_entropy(probabilities):
    """Computes the total entropy of a motif given the probabilities."""
    # Ensure probabilities are within (1e-6, 1 - 1e-6)
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    entropy_per_position = - \
        np.sum(probabilities * np.log2(probabilities), axis=1)
    total_entropy = np.sum(entropy_per_position)
    return total_entropy


def is_valid_pwm(pwm):
    """Checks if a PWM or probabilities matrix contains any NaNs or infinite values."""
    return not (np.isnan(pwm).any() or np.isinf(pwm).any())
