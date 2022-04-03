import numpy as np


# spearman coefficient rs
def spearman(R, Q):
    """
    Calculate Spearman rank correlation coefficient between two vectors
    Parameters
    -----------
        R : ndarray
            First vector containing values
        Q : ndarray
            Second vector containing values
    Returns
    --------
        float
            Value of correlation coefficient between two vectors

    Examples
    ----------
    >>> rS = spearman(R, Q)
    """
    N = len(R)
    denominator = N*(N**2-1)
    numerator = 6*sum((R-Q)**2)
    rS = 1-(numerator/denominator)
    return rS


# weighted spearman coefficient rw
def weighted_spearman(R, Q):
    """
    Calculate Weighted Spearman rank correlation coefficient between two vectors
    Parameters
    -----------
        R : ndarray
            First vector containing values
        Q : ndarray
            Second vector containing values
    Returns
    --------
        float
            Value of correlation coefficient between two vectors

    Examples
    ---------
    >>> rW = weighted_spearman(R, Q)
    """
    N = len(R)
    denominator = N**4 + N**3 - N**2 - N
    numerator = 6 * sum((R - Q)**2 * ((N - R + 1) + (N - Q + 1)))
    rW = 1 - (numerator / denominator)
    return rW


# pearson coefficient
def pearson_coeff(R, Q):
    """
    Calculate Pearson correlation coefficient between two vectors
    Parameters
    -----------
        R : ndarray
            First vector containing values
        Q : ndarray
            Second vector containing values
    Returns
    --------
        float
            Value of correlation coefficient between two vectors

    Examples
    ----------
    >>> corr = pearson_coeff(R, Q)
    """
    numerator = np.sum((R - np.mean(R)) * (Q - np.mean(Q)))
    denominator = np.sqrt(np.sum((R - np.mean(R))**2) * np.sum((Q - np.mean(Q))**2))
    corr = numerator / denominator
    return corr


# rank similarity coefficient WS
def WS_coeff(R, Q):
    """
    Calculate Rank smilarity coefficient between two vectors

    Parameters
    -----------
        R : ndarray
            First vector containing values
        Q : ndarray
            Second vector containing values

    Returns
    --------
        float
            Value of similarity coefficient between two vectors

    Examples
    ----------
    >>> ws = WS_coeff(R, Q)
    """
    N = len(R)
    numerator = 2**(-np.float64(R)) * np.abs(R - Q)
    denominator = np.max((np.abs(R - 1), np.abs(R - N)), axis = 0)
    ws = 1 - np.sum(numerator / denominator)
    return ws