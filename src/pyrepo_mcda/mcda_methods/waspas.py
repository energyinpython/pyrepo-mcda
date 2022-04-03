import numpy as np
from ..normalizations import linear_normalization
from .mcda_method import MCDA_method


class WASPAS(MCDA_method):
    def __init__(self, normalization_method = linear_normalization, lambda_param = 0.5):
        """
        Create the WASPAS method object and select normalization method `normalization_method`, default
        normalization method for WASPAS is `linear_normalization` and lambda parameter `lambda_param`, 
        which is set on 0.5 by default.

        Parameters
        -----------
            normalization_method : function
                method for decision matrix normalization chosen from `normalizations`

            lambda_param : float
                lambda parameter is between 0 and 1
        """

        self.normalization_method = normalization_method
        self.lambda_param = lambda_param


    def __call__(self, matrix, weights, types):
        """
        Score alternatives provided in decision matrix `matrix` with m alternatives and n criteria 
        using criteria `weights` and criteria `types`.

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
        ----------
        >>> waspas = WASPAS(normalization_method = linear_normalization, lambda_param = 0.5)
        >>> pref = waspas(matrix, weights, types)
        >>> rank = rank_preferences(pref, reverse = True)
        """
        WASPAS._verify_input_data(matrix, weights, types)
        return WASPAS._waspas(matrix, weights, types, self.normalization_method, self.lambda_param)


    @staticmethod
    def _waspas(matrix, weights, types, normalization_method, lambda_param):
        # Normalize decision matrix
        norm_matrix = normalization_method(matrix, types)
        # Calculate the total relative importance of alternatives based on WSM
        Q1 = np.sum((norm_matrix * weights), axis = 1)
        # Calculate the total relative importance of alternatives based on WPM
        Q2 = np.prod((norm_matrix ** weights), axis = 1)
        # Determine the total relative importance of alternatives
        # If lambda is equal to 0, WASPAS method is transformed to WPM
        # If lambda is equal to 1, WASPAS becomes WSM
        Q = lambda_param * Q1 + (1 - lambda_param) * Q2
        return Q
