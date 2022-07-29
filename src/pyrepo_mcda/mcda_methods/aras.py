import numpy as np

from .mcda_method import MCDA_method
from ..normalizations import sum_normalization



class ARAS(MCDA_method):
    def __init__(self, normalization_method = sum_normalization):
        """
        Create the ARAS method object
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
                Criteria weights. Sum of weights must be equal to 1.
            types: ndarray
                Criteria types. Profit criteria are represented by 1 and cost by -1.

        Returns
        --------
            ndrarray
                Preference values of each alternative. The best alternative has the highest preference value. 

        Examples
        ----------
        >>> aras = ARAS()
        >>> pref = aras(matrix, weights, types)
        >>> rank = rank_preferences(pref, reverse = True)
        """
        
        ARAS._verify_input_data(matrix, weights, types)
        return ARAS._aras(matrix, weights, types, self.normalization_method)

    @staticmethod
    def _aras(matrix, weights, types, normalization_method):
        # Create optimal alternative
        A0 = np.zeros(matrix.shape[1])
        A0[types == 1] = np.max(matrix[:, types == 1], axis = 0)
        A0[types == -1] = np.min(matrix[:, types == -1], axis = 0)
        matrix = np.vstack((A0, matrix))
        # Normalize matrix using the sum normalization method
        norm_matrix = normalization_method(matrix, types)
        # Calculate the weighted normalized decision matrix
        d = norm_matrix * weights
        # Calculate the optimality function for each alternative
        S = np.sum(d, axis = 1)
        # Determine the degree of utility
        U = S / S[0]
        return U[1:]