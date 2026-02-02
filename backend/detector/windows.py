"""
Sliding window management for UCC detection.
Based on SPEC.md Section 6.2.
"""
import numpy as np
from typing import List, Tuple


def get_window_indices(t: int, k: int, total: int, mode: str = "minus") -> List[int]:
    """
    Get window indices for position t.

    Args:
        t: Current position
        k: Window size (k_minus for past, k_plus for current)
        total: Total number of segments
        mode: "minus" for past window, "plus" for current window

    Returns:
        List of indices in the window
    """
    if mode == "minus":
        # Past window: [t-k, ..., t-1]
        start = max(0, t - k)
        end = t
    elif mode == "plus":
        # Current window: [t, ..., t+k-1]
        start = t
        end = min(total, t + k)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'minus' or 'plus'")

    return list(range(start, end))


def compute_context_embedding(
    embeddings: np.ndarray,
    indices: List[int]
) -> np.ndarray:
    """
    Compute context embedding as mean of embeddings in window.

    Args:
        embeddings: All embeddings (N, d)
        indices: Window indices

    Returns:
        Mean embedding vector (d,)
    """
    if not indices:
        # Return zero vector if empty
        return np.zeros(embeddings.shape[1])

    window_embeddings = embeddings[indices]
    return np.mean(window_embeddings, axis=0)


def compute_all_context_embeddings(
    embeddings: np.ndarray,
    k_minus: int,
    k_plus: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Compute all context embeddings for the entire sequence.

    Args:
        embeddings: All embeddings (N, d)
        k_minus: Past window size
        k_plus: Current window size

    Returns:
        Tuple of (c_minus_list, c_plus_list)
    """
    N = embeddings.shape[0]
    c_minus_list = []
    c_plus_list = []

    for t in range(N):
        # Past window with k_minus
        minus_indices = get_window_indices(t, k_minus, N, mode="minus")
        c_minus = compute_context_embedding(embeddings, minus_indices)
        c_minus_list.append(c_minus)

        # Current window with k_plus
        plus_indices = get_window_indices(t, k_plus, N, mode="plus")
        c_plus = compute_context_embedding(embeddings, plus_indices)
        c_plus_list.append(c_plus)

    return c_minus_list, c_plus_list
