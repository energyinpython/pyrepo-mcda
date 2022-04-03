import numpy as np

from ..normalizations import minmax_normalization
from ..distance_metrics import euclidean
from .mcda_method import MCDA_method

class TOPSIS(MCDA_method):
    def __init__(self, normalization_method = minmax_normalization, distance_metric = euclidean):
        """
        Create the TOPSIS method object and select normalization method `normalization_method` and
        distance metric `distance metric`.

        Parameters
        -----------
            normalization_method : function
                method for decision matrix normalization chosen from `normalizations`

            distance_metric : functions
                method for calculating the distance between two vectors
        """
        self.normalization_method = normalization_method
        self.distance_metric = distance_metric


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
        >>> topsis = TOPSIS(normalization_method = minmax_normalization, distance_metric = euclidean)
        >>> pref = topsis(matrix, weights, types)
        >>> rank = rank_preferences(pref, reverse = True)
        """
        TOPSIS._verify_input_data(matrix, weights, types)
        return TOPSIS._topsis(matrix, weights, types, self.normalization_method, self.distance_metric)


    @staticmethod
    def _topsis(matrix, weights, types, normalization_method, distance_metric):
        # Normalize matrix using chosen normalization (for example linear normalization)
        norm_matrix = normalization_method(matrix, types)

        # Multiply all rows of normalized matrix by weights
        weighted_matrix = norm_matrix * weights

        # Calculate vectors of PIS (ideal solution) and NIS (anti-ideal solution)
        pis = np.max(weighted_matrix, axis=0)
        nis = np.min(weighted_matrix, axis=0)

        # Calculate chosen distance of every alternative from PIS and NIS using chosen distance metric `distance_metric` from `distance_metrics`
        Dp = np.array([distance_metric(x, pis) for x in weighted_matrix])
        Dm = np.array([distance_metric(x, nis) for x in weighted_matrix])

        C = Dm / (Dm + Dp)
        return C