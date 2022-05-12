import copy
import itertools
import numpy as np


# Euclidean distance
def euclidean(A, B):
    """
    Calculate Euclidean distance between two vectors `A` and `B`.

    Parameters
    -----------
        A : ndarray
            First vector containing values
        B : ndarray
            Second vector containing values

    Returns
    --------
        float
            distance value between two vectors

    Examples
    ----------
    >>> distance = euclidean(A, B)
    """

    tmp = np.sqrt(np.sum(np.square(A - B)))
    return tmp

# Manhattan distance
def manhattan(A, B):
    """
    Calculate Manhattan (Taxicab) distance between two vectors `A` and `B`.

    Parameters
    -----------
        A : ndarray
            First vector containing values
        B : ndarray
            Second vector containing values

    Returns
    --------
        float
            distance value between two vectors

    Examples
    ----------
    >>> distance = manhattan(A, B)
    """

    tmp = np.sum(np.abs(A - B))
    return tmp


# function for calculation Hausdorff distance using `hausdorff` function
def hausdorff_distance(A, B):
    min_h = np.inf
    for i, j in itertools.product(range(len(A)), range(len(B))):
        d = euclidean(A[i], B[j])
        if d < min_h:
            min_h = d
            min_ind = j

    max_h = -np.inf
    for i in range(len(A)):
        d = euclidean(A[i], B[min_ind])
        if d > max_h:
            max_h = d

    return max_h


# Hausdorff distance
def hausdorff(A, B):
    """
    Calculate Hausdorff distance between two vectors `A` and `B`.

    Parameters
    -----------
        A : ndarray
            First vector containing values
        B : ndarray
            Second vector containing values

    Returns
    --------
        float
            distance value between two vectors

    Examples
    ----------
    >>> distance = hausdorff(A, B)
    """

    ah = hausdorff_distance(A, B)
    bh = hausdorff_distance(B, A)
    tmp = max(ah, bh)
    return tmp


# correlation distance
def correlation(A, B):
    """
    Calculate Correlation distance between two vectors `A` and `B`.

    Parameters
    -----------
        A : ndarray
            First vector containing values
        B : ndarray
            Second vector containing values

    Returns
    --------
        float
            distance value between two vectors

    Examples
    ----------
    >>> distance = correlation(A, B)
    """

    numerator = np.sum((A - np.mean(A)) * (B - np.mean(B)))
    denominator = np.sqrt(np.sum((A - np.mean(A)) ** 2)) * np.sqrt(np.sum((B - np.mean(B)) ** 2))
    if denominator == 0:
        denominator = 1
    tmp = 1 - (numerator / denominator)
    return tmp


# Chebyshev distance
def chebyshev(A, B):
    """
    Calculate Chebyshev distance between two vectors `A` and `B`.

    Parameters
    -----------
        A : ndarray
            First vector containing values
        B : ndarray
            Second vector containing values

    Returns
    --------
        float
            distance value between two vectors

    Examples
    ----------
    >>> distance = chebyshev(A, B)
    """

    max_h = -np.inf
    for i, j in itertools.product(range(len(A)), range(len(B))):
        d = np.abs(A[i] - B[j])
        if d > max_h:
            max_h = d

    return max_h


# standardized Euclidean distance
def std_euclidean(A, B):
    """
    Calculate Standardized Euclidean distance between two vectors `A` and `B`.

    Parameters
    -----------
        A : ndarray
            First vector containing values
        B : ndarray
            Second vector containing values

    Returns
    --------
        float
            distance value between two vectors

    Examples
    ----------
    >>> distance = std_euclidean(A, B)
    """

    tab_std = np.vstack((A, B))
    stdv = np.sum(np.square(tab_std - np.mean(tab_std, axis = 0)), axis = 0)
    stdv = np.sqrt(stdv / tab_std.shape[0])
    stdv[stdv == 0] = 1
    tmp = np.sum(np.square((A - B) / stdv))
    tmp = np.sqrt(tmp)
    return tmp


# cosine distance
def cosine(A, B):
    """
    Calculate Cosine distance between two vectors `A` and `B`.

    Parameters
    -----------
        A : ndarray
            First vector containing values
        B : ndarray
            Second vector containing values

    Returns
    --------
        float
            distance value between two vectors

    Examples
    ----------
    >>> distance = cosine(A, B)
    """

    numerator = np.sum(A * B)
    denominator = (np.sqrt(np.sum(np.square(A)))) * (np.sqrt(np.sum(np.square(B))))
    if denominator == 0:
        denominator = 1
    tmp = 1 - (numerator / denominator)
    return tmp


# cosine similarity measure
def csm(A, B):
    """
    Calculate Cosine similarity measure of distance between two vectors `A` and `B`.

    Parameters
    -----------
        A : ndarray
            First vector containing values
        B : ndarray
            Second vector containing values

    Returns
    --------
        float
            distance value between two vectors

    Examples
    ---------
    >>> distance = csm(A, B)
    """

    numerator = np.sum(A * B)
    denominator = (np.sqrt(np.sum(A))) * (np.sqrt(np.sum(B)))
    if denominator == 0:
        denominator = 1
    tmp = numerator / denominator
    return tmp


# squared Euclidean distance
def squared_euclidean(A, B):
    """
    Calculate Squared Euclidean distance between two vectors `A` and `B`.

    Parameters
    -----------
        A : ndarray
            First vector containing values
        B : ndarray
            Second vector containing values

    Returns
    --------
        float
            distance value between two vectors

    Examples
    ----------
    >>> distance = squared_euclidean(A, B)
    """

    tmp = np.sum(np.square(A - B))
    return tmp


# Sorensen or Bray-curtis distance
def bray_curtis(A, B):
    """
    Calculate Bray-Curtis distance between two vectors `A` and `B`.

    Parameters
    -----------
        A : ndarray
            First vector containing values
        B : ndarray
            Second vector containing values

    Returns
    --------
        float
            distance value between two vectors

    Examples
    ----------
    >>> distance = bray_curtis(A, B)
    """

    numerator = np.sum(np.abs(A - B))
    denominator = np.sum(A + B)
    if denominator == 0:
        denominator = 1
    tmp = numerator / denominator
    return tmp


# Canberra distance
def canberra(A, B):
    """
    Calculate Canberra distance between two vectors `A` and `B`.

    Parameters
    -----------
        A : ndarray
            First vector containing values
        B : ndarray
            Second vector containing values

    Returns
    --------
        float
            distance value between two vectors

    Examples
    ----------
    >>> distance = canberra(A, B)
    """

    numerator = np.abs(A - B)
    denominator = A + B
    denominator[denominator == 0] = 1
    tmp = np.sum(numerator / denominator)
    return tmp


# Lorentzian distance
def lorentzian(A, B):
    """
    Calculate Lorentzian distance between two vectors `A` and `B`.

    Parameters
    -----------
        A : ndarray
            First vector containing values
        B : ndarray
            Second vector containing values

    Returns
    --------
        float
            distance value between two vectors

    Examples
    ----------
    >>> distance = lorentzian(A, B)
    """

    tmp = np.sum(np.log(1 + np.abs(A - B)))
    return tmp


# Jaccard distance
def jaccard(A, B):
    """
    Calculate Jaccard distance between two vectors `A` and `B`.

    Parameters
    -----------
        A : ndarray
            First vector containing values
        B : ndarray
            Second vector containing values

    Returns
    --------
        float
            distance value between two vectors

    Examples
    -----------
    >>> distance = jaccard(A, B)
    """

    numerator = np.sum(np.square(A - B))
    denominator = np.sum(A ** 2) + np.sum(B ** 2) - np.sum(A * B)
    if denominator == 0:
        denominator = 1
    tmp = numerator / denominator
    return tmp


# Dice distance
def dice(A, B):
    """
    Calculate Dice distance between two vectors `A` and `B`.

    Parameters
    -----------
        A : ndarray
            First vector containing values
        B : ndarray
            Second vector containing values

    Returns
    --------
        float
            distance value between two vectors

    Examples
    ----------
    >>> distance = dice(A, B)
    """

    numerator = np.sum(np.square(A - B))
    denominator = np.sum(A ** 2) + np.sum(B ** 2)
    if denominator == 0:
        denominator = 1
    tmp = numerator / denominator
    return tmp


# Bhattacharyya distance
def bhattacharyya(A, B):
    """
    Calculate Bhattacharyya distance between two vectors `A` and `B`.

    Parameters
    -----------
        A : ndarray
            First vector containing values
        B : ndarray
            Second vector containing values

    Returns
    --------
        float
            distance value between two vectors

    Examples
    ---------
    >>> distance = bhattacharyya(A, B)
    """

    value = (np.sum(np.sqrt(A * B)))**2
    if value == 0:
        tmp = 0
    else:
        tmp = -np.log(value)
    return tmp


# Hellinger distance
def hellinger(A, B):
    """
    Calculate Hellinger distance between two vectors `A` and `B`.

    Parameters
    -----------
        A : ndarray
            First vector containing values
        B : ndarray
            Second vector containing values

    Returns
    --------
        float
            distance value between two vectors

    Examples
    -----------
    >>> distance = hellinger(A, B)
    """

    value = 1 - np.sum(np.sqrt(A * B))
    if value < 0:
        value = 0
    tmp = 2 * np.sqrt(value)
    return tmp


# Matusita distance
def matusita(A, B):
    """
    Calculate Matusita distance between two vectors `A` and `B`.

    Parameters
    -----------
        A : ndarray
            First vector containing values
        B : ndarray
            Second vector containing values

    Returns
    --------
        float
            distance value between two vectors

    Examples
    ----------
    >>> distance = matusita(A, B)
    """

    value = 2 - 2 * (np.sum(np.sqrt(A * B)))
    if value < 0:
        value = 0
    tmp = np.sqrt(value)
    return tmp


# squared-chord distance
def squared_chord(A, B):
    """
    Calculate Squared-Chord distance between two vectors `A` and `B`.

    Parameters
    -----------
        A : ndarray
            First vector containing values
        B : ndarray
            Second vector containing values

    Returns
    --------
        float
            distance value between two vectors

    Examples
    ---------
    >>> distance = squared_chord(A, B)
    """

    tmp = np.sum(np.square(np.sqrt(A) - np.sqrt(B)))
    return tmp


# Pearson chi-square distance
def pearson_chi_square(A, B):
    """
    Calculate Pearson Chi Square distance between two vectors `A` and `B`.

    Parameters
    -----------
        A : ndarray
            First vector containing values
        B : ndarray
            Second vector containing values

    Returns
    --------
        float
            distance value between two vectors

    Examples
    ---------
    >>> distance = pearson_chi_square(A, B)
    """

    numerator = np.square(A - B)
    denominator = copy.deepcopy(B)
    denominator[denominator == 0] = 1
    tmp = np.sum(numerator / denominator)
    return tmp

# squared chi-square distance
def squared_chi_square(A, B):
    """
    Calculate Squared Chi Sqaure distance between two vectors `A` and `B`.

    Parameters
    -----------
        A : ndarray
            First vector containing values
        B : ndarray
            Second vector containing values

    Returns
    --------
        float
            distance value between two vectors

    Examples
    ----------
    >>> distance = squared_chi_square(A, B)
    """
    
    numerator = np.square(A - B)
    denominator = A + B
    denominator[denominator == 0] = 1
    tmp = np.sum(numerator / denominator)
    return tmp