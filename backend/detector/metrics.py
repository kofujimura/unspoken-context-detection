"""
Core metrics for UCC detection.
Based on SPEC.md Section 6.
"""
import numpy as np
from typing import Optional


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity in range [-1, 1]
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def context_debt(c_minus: np.ndarray, c_plus: np.ndarray) -> float:
    """
    Compute Context Debt (proxy for implicit prerequisite load).

    D(t) = ||c_plus(t) - c_minus(t)||_2

    Args:
        c_minus: Past context embedding
        c_plus: Current context embedding

    Returns:
        Context debt value
    """
    diff = c_plus - c_minus
    return float(np.linalg.norm(diff))


def sigmoid(x: float) -> float:
    """
    Sigmoid function.

    Args:
        x: Input value

    Returns:
        Sigmoid output in range (0, 1)
    """
    return 1.0 / (1.0 + np.exp(-x))


def normalize_scores_minmax(scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalization to [0, 1].

    Args:
        scores: Input scores

    Returns:
        Normalized scores
    """
    min_val = np.min(scores)
    max_val = np.max(scores)

    if max_val == min_val:
        return np.zeros_like(scores)

    return (scores - min_val) / (max_val - min_val)


def normalize_scores_zsigmoid(scores: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Z-score sigmoid normalization (recommended).

    ucc_score = sigmoid((score - mu) / sigma)

    Args:
        scores: Input scores
        mu: Mean of scores
        sigma: Standard deviation of scores

    Returns:
        Normalized scores in range (0, 1)
    """
    if sigma == 0:
        return np.full_like(scores, 0.5)

    z_scores = (scores - mu) / sigma
    return np.array([sigmoid(z) for z in z_scores])
