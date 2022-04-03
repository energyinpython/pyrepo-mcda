import numpy as np
from ..normalizations import minmax_normalization
from .mcda_method import MCDA_method


class MABAC(MCDA_method):
    def __init__(self, normalization_method = minmax_normalization):
        """
        Create the MABAC method object and select normalization method `normalization_method` from
        `normalizations`. The default normalization method for MABAC method is `minmax_normalization`

        Parameters
        -----------
            normalization_method : function
                method for decision matrix normalization chosen from `normalizations`
        """
        self.normalization_method = normalization_method


    def __call__(self, matrix, weights, types):
        """
        Score alternatives provided in decision matrix `matrix` using criteria `weights` and criteria `types`.

        Parameters
        -----------
            matrix : ndarray
                Decision matrix with m alternatives in rows and n criteria in columns.
            weights: ndarray
                Vector with criteria weights. Sum of weights must be equal to 1.
            types: ndarray
                Vector with criteria types. Profit criteria are represented by 1 and cost by -1.

        Returns
        --------
            ndrarray
                Vector with preference values of each alternative. The best alternative has the highest preference value. 

        Examples
        ---------
        >>> mabac = MABAC(normalization_method = minmax_normalization)
        >>> pref = mabac(matrix, weights, types)
        >>> rank = rank_preferences(pref, reverse = True)
        """
        MABAC._verify_input_data(matrix, weights, types)
        return MABAC._mabac(matrix, weights, types, self.normalization_method)


    def _mabac(matrix, weights, types, normalization_method):
        m, n = matrix.shape
        # Normalize the decision matrix
        norm_matrix = normalization_method(matrix, types)
        # Calculate elements from the weighted matrix
        V = weights * (norm_matrix + 1)
        # Determine the border approximation area matrix
        G = np.product(V, axis = 0) ** (1/m)
        # Calculate distance of alternatives from the border approximation area for the matrix elements
        Q = V - G
        # Calculate the sum of distance of alternatives from the border approximation areas
        # By calculating the sum of elements from Q matrix by rows we obtain the final values of
        # the criterion functions of the alternatives
        S = np.sum(Q, axis = 1)
        return S

