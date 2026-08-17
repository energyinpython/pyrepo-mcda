import numpy as np


# linear normalization
def linear_normalization(matrix, types):
    """
    Normalize decision matrix using linear normalization method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with m alternatives in rows and n criteria in columns
        types : ndarray
            Criteria types. Profit criteria are represented by 1 and cost by -1.

    Returns
    --------
        ndarray
            Normalized decision matrix

    Examples
    ----------
    >>> nmatrix = linear_normalization(matrix, types)
    """
    x_norm = np.zeros(np.shape(matrix))
    x_norm[:, types == 1] = matrix[:, types == 1] / (np.amax(matrix[:, types == 1], axis = 0))
    x_norm[:, types == -1] = np.amin(matrix[:, types == -1], axis = 0) / matrix[:, types == -1]
    return x_norm


# min-max normalization
def minmax_normalization(matrix, types):
    """
    Normalize decision matrix using minimum-maximum normalization method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with m alternatives in rows and n criteria in columns
        types : ndarray
            Criteria types. Profit criteria are represented by 1 and cost by -1.

    Returns
    --------
        ndarray
            Normalized decision matrix

    Examples
    ----------
    >>> nmatrix = minmax_normalization(matrix, types)
    """
    matrix = np.asarray(matrix, dtype=float)
    x_norm = np.ones_like(matrix)

    # Profit criteria
    profit = types == 1
    if np.any(profit):
        mins = matrix[:, profit].min(axis=0)
        maxs = matrix[:, profit].max(axis=0)
        ranges = maxs - mins

        x_norm[:, profit] = np.divide(
            matrix[:, profit] - mins,
            ranges,
            out=np.ones_like(matrix[:, profit]),
            where=ranges != 0
        )

    # Cost criteria
    cost = types == -1
    if np.any(cost):
        mins = matrix[:, cost].min(axis=0)
        maxs = matrix[:, cost].max(axis=0)
        ranges = maxs - mins

        x_norm[:, cost] = np.divide(
            maxs - matrix[:, cost],
            ranges,
            out=np.ones_like(matrix[:, cost]),
            where=ranges != 0
        )

    return x_norm


# max normalization
def max_normalization(matrix, types):
    """
    Normalize decision matrix using maximum normalization method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with m alternatives in rows and n criteria in columns
        types : ndarray
            Criteria types. Profit criteria are represented by 1 and cost by -1.

    Returns
    --------
        ndarray
            Normalized decision matrix

    Examples
    ----------
    >>> nmatrix = max_normalization(matrix, types)
    """
    maximes = np.amax(matrix, axis = 0)
    matrix = matrix / maximes
    matrix[:, types == -1] = 1 - matrix[:, types == -1]
    return matrix


# sum normalization
def sum_normalization(matrix, types):
    """
    Normalize decision matrix using sum normalization method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with m alternatives in rows and n criteria in columns
        types : ndarray
            Criteria types. Profit criteria are represented by 1 and cost by -1.

    Returns
    --------
        ndarray
            Normalized decision matrix

    Examples
    ----------
    >>> nmatrix = sum_normalization(matrix, types)
    """
    x_norm = np.zeros((matrix.shape[0], matrix.shape[1]))
    x_norm[:, types == 1] = matrix[:, types == 1] / np.sum(matrix[:, types == 1], axis = 0)
    x_norm[:, types == -1] = (1 / matrix[:, types == -1]) / np.sum((1 / matrix[:, types == -1]), axis = 0)

    return x_norm


# vector normalization
def vector_normalization(matrix, types):
    """
    Normalize decision matrix using vector normalization method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with m alternatives in rows and n criteria in columns
        types : ndarray
            Criteria types. Profit criteria are represented by 1 and cost by -1.

    Returns
    --------
        ndarray
            Normalized decision matrix

    Examples
    -----------
    >>> nmatrix = vector_normalization(matrix, types)
    """
    x_norm = np.zeros((matrix.shape[0], matrix.shape[1]))
    x_norm[:, types == 1] = matrix[:, types == 1] / (np.sum(matrix[:, types == 1] ** 2, axis = 0))**(0.5)
    x_norm[:, types == -1] = 1 - (matrix[:, types == -1] / (np.sum(matrix[:, types == -1] ** 2, axis = 0))**(0.5))

    return x_norm


# multimoora normalization
def multimoora_normalization(matrix):
    """
    Normalize decision matrix using vector normalization method as for profit criteria.

    Parameters
    ------------
        matrix : ndarray
            Decision matrix with m alternatives in rows and n criteria in columns

    Examples
    -----------
    >>> nmatrix = multimoora_normalization(matrix)
    """
    x_norm = matrix / ((np.sum(matrix ** 2, axis = 0))**(0.5))
    return x_norm
