"""
Computes the cosine-similarities for LSA.
"""

import numpy as np


def cos_sim(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Compute the cosine similarity (cosine angle) between two vectors.

    :param vector1: the first vector (np.ndarray).
    :param vector2: the second vector (np.ndarray).
    :return: the cosine which is between -1 and 1.
    """
    # Use dot product to compute cos angle.
    # cos(x) = u·v / (|u||v|)
    dot = np.dot(vector1, vector2)
    len1 = np.linalg.norm(vector1)  # norm computes the length of the vector.
    len2 = np.linalg.norm(vector2)
    return dot / (len1 * len2)


def norm_avg_cos_sims(values):
    """
    Compute the normalized average of a list (or iterable) of cosine similarity values.
    :param values: a list (or iterable) of values cosines. The values are assumed to be [-1, 1].
    :return: a float of the normalized average [0, 1].
    """
    return (np.average(values) + 1) / 2


def norm_std_cos_sims(values) -> float:
    """
    Compute the normalized standard deviation of a list (or iterable) of cosine similarity values.
    :param values: a list (or iterable) of values cosines. The values are assumed to be [-1, 1].
    :return: a float of the normalized standard deviation [0, 1].
    """
    return (np.std(values) + 1) / 2
