import numpy as np
import itertools

from .correlations import pearson_coeff
from .normalizations import sum_normalization, minmax_normalization


# equal weighting
def equal_weighting(matrix):
    """
    Calculate criteria weights using objective Equal weighting method
    Parameters
    ----------
        matrix : ndarray
            Decision matrix with performance values of m alternatives and n criteria

    Returns
    -------
        ndarray
            vector of criteria weights
    """
    N = np.shape(matrix)[1]
    return np.ones(N) / N


# Entropy weighting
def entropy_weighting(matrix):
    """
    Calculate criteria weights using objective Entropy weighting method
    Parameters
    ----------
        matrix : ndarray
            Decision matrix with performance values of m alternatives and n criteria

    Returns
    -------
        ndarray
            vector of criteria weights
    """
    # normalize the decision matrix with sum_normalization method from normalizations as for profit criteria
    types = np.ones(np.shape(matrix)[1])
    pij = sum_normalization(matrix, types)
    # Transform negative values in decision matrix `matrix` to positive values
    pij = np.abs(pij)
    m, n = np.shape(pij)
    H = np.zeros((m, n))

    # Calculate entropy
    for j, i in itertools.product(range(n), range(m)):
        if pij[i, j]:
            H[i, j] = pij[i, j] * np.log(pij[i, j])

    h = np.sum(H, axis = 0) * (-1 * ((np.log(m)) ** (-1)))

    # Calculate degree of diversification
    d = 1 - h

    # Set w as the degree of importance of each criterion
    w = d / (np.sum(d))
    return w


# Standard Deviation weighting
def std_weighting(matrix):
    """
    Calculate criteria weights using objective Standard deviation weighting method
    Parameters
    ----------
        matrix : ndarray
            Decision matrix with performance values of m alternatives and n criteria
            
    Returns
    -------
        ndarray
            vector of criteria weights
    """

    # Calculate the standard deviation of each criterion in decision matrix
    stdv = np.sqrt((np.sum(np.square(matrix - np.mean(matrix, axis = 0)), axis = 0)) / (matrix.shape[0]))
    # Calculate criteria weights by dividing the standard deviations by their sum
    return stdv / np.sum(stdv)


# CRITIC weighting
def critic_weighting(matrix):
    """
    Calculate criteria weights using objective CRITIC weighting method
    Parameters
    ----------
        matrix : ndarray
            Decision matrix with performance values of m alternatives and n criteria
            
    Returns
    -------
        ndarray
            vector of criteria weights
    """
    # Normalize the decision matrix using Minimum-Maximum normalization minmax_normalization from normalizations as for profit criteria
    types = np.ones(np.shape(matrix)[1])
    x_norm = minmax_normalization(matrix, types)
    # Calculate the standard deviation
    std = np.std(x_norm, axis = 0)
    n = np.shape(x_norm)[1]
    # Calculate correlation coefficients of all pairs of columns of normalized decision matrix
    correlations = np.zeros((n, n))
    for i, j in itertools.product(range(n), range(n)):
        correlations[i, j] = pearson_coeff(x_norm[:, i], x_norm[:, j])

    # Calculate the difference between 1 and calculated correlations
    difference = 1 - correlations
    # Multiply the difference by the standard deviation
    C = std * np.sum(difference, axis = 0)
    # Calculate the weights by dividing vector with C by their sum
    w = C / np.sum(C)
    return w