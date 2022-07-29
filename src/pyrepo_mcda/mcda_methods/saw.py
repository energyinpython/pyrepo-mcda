import numpy as np

from .mcda_method import MCDA_method
from ..normalizations import linear_normalization



class SAW(MCDA_method):
    def __init__(self, normalization_method = linear_normalization):
        """
        Create the SAW method object and select normalization method `normalization_method`.

        Parameters
        -----------
            normalization_method : function
                method for decision matrix normalization chosen from `normalizations`
        """
        
        self.normalization_method = normalization_method

    def __call__(self, matrix, weights, types):
        """
        Score alternatives provided in decision matrix `matrix` with m alternatives in rows and 
        n criteria in columns using criteria `weights` and criteria `types`.

        Parameters
        ----------
            matrix : ndarray
                Decision matrix with m alternatives in rows and n criteria in columns.
            weights: ndarray
                Vector with criteria weights. Sum of weights must be equal to 1.
            types: ndarray
                Vector with criteria types. Profit criteria are represented by 1 and cost by -1.

        Returns
        -------
            ndrarray
                Vector with preference values of each alternative. The best alternative has the highest preference value. 
        
        Examples
        ---------
        >>> saw = SAW(normalization_method = minmax_normalization)
        >>> pref = saw(matrix, weights, types)
        >>> rank = rank_preferences(pref, reverse = True)
        """

        SAW._verify_input_data(matrix, weights, types)
        return SAW._saw(matrix, weights, types, self.normalization_method)


    @staticmethod
    def _saw(matrix, weights, types, normalization_method):
        # Normalize matrix using chosen normalization (for example linear normalization)
        norm_matrix = normalization_method(matrix, types)

        # Multiply all rows of normalized matrix by weights
        weighted_matrix = norm_matrix * weights

        # Aggregate and return scores
        return np.sum(weighted_matrix, axis = 1)