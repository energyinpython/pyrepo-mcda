import numpy as np
from ..normalizations import sum_normalization
from .mcda_method import MCDA_method


class COPRAS(MCDA_method):
    def __init__(self, normalization_method = sum_normalization):
        """
        Create the COPRAS method object and select normalization method `normalization_method`

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
                Criteria weights. Sum of weights must be equal to 1.
            types: ndarray
                Criteria types. Profit criteria are represented by 1 and cost by -1.

        Returns
        --------
            ndrarray
                Preference values of each alternative. The best alternative has the highest preference value. 

        Examples
        ----------
        >>> copras = COPRAS(normalization_method = sum_normalization)
        >>> pref = copras(matrix, weights, types)
        >>> rank = rank_preferences(pref, reverse = True)
        """

        COPRAS._verify_input_data(matrix, weights, types)
        return COPRAS._copras(matrix, weights, types, self.normalization_method)

    @staticmethod
    def _copras(matrix, weights, types, normalization_method):
        # Normalize matrix as for profit criteria using chosen normalization method.
        # norm_matrix = matrix/np.sum(matrix, axis = 0)
        norm_matrix = normalization_method(matrix, np.ones(len(weights)))
        # Multiply all values in the normalized matrix by weights.
        d = norm_matrix * weights
        # Calculate the sums of weighted normalized outcomes for profit criteria.
        Sp = np.sum(d[:, types == 1], axis = 1)
        # Calculate the sums of weighted normalized outcomes for cost criteria.
        Sm = np.sum(d[:, types == -1], axis = 1)
        # Calculate the relative priority Q of evaluated options.
        Q = Sp + ((np.sum(Sm))/(Sm * np.sum(1 / Sm)))
        # Calculate the quantitive utility value for each of the evaluated options.
        U = Q / np.max(Q)
        return U


