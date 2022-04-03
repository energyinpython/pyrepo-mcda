import itertools
import numpy as np

from ..normalizations import linear_normalization
from ..distance_metrics import euclidean
from .mcda_method import MCDA_method


class CODAS(MCDA_method):
    def __init__(self, normalization_method = linear_normalization, distance_metric = euclidean, tau = 0.02):
        """
        Create the CODAS method object and select normalization method `normalization_method`, default
        normalization method for CODAS is `linear_normalization`, distance metric 
        `distance_metric` selected from `distance_metrics`, which is `euclidean` by default and tau parameter `tau`, 
        which is set on 0.02 by default.

        Parameters
        -----------
            normalization_method : function
                method for decision matrix normalization chosen from `normalizations`

            distance_metric : functions
                method for calculating the distance between two vectors

            tau : float
                the threshold parameter between 0.01 to 0.05. If the difference between 
                Euclidean `euclidean` or other selected distances of two alternatives is less than tau, these two alternatives 
                are also compared by the Taxicab distance
        """
        self.normalization_method = normalization_method
        self.distance_metric = distance_metric
        self.tau = tau


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
        >>> codas = CODAS(normalization_method = linear_normalization, distance_metric = euclidean, tau = 0.02)
        >>> pref = codas(matrix, weights, types)
        >>> rank = rank_preferences(pref, reverse = True)
        """
        CODAS._verify_input_data(matrix, weights, types)
        return CODAS._codas(self, matrix, weights, types, self.normalization_method, self.distance_metric)


    # psi 0.01 - 0.05 recommended range of tau (threshold parameter) value
    def _psi(self, x):
        return 1 if np.abs(x) >= self.tau else 0


    @staticmethod
    def _codas(self, matrix, weights, types, normalization_method, distance_metric):
        # Normalize matrix using linear normalization
        norm_matrix = normalization_method(matrix, types)
    
        # Multiply all rows of normalized matrix by weights
        weighted_matrix = norm_matrix * weights
        m, n = weighted_matrix.shape

        # Calculate NIS vector (anti-ideal solution)
        nis = np.min(weighted_matrix, axis=0)

        # Calculate chosen distance (for example Euclidean) and Taxicab distance from anti-ideal solution
        E = np.array([distance_metric(x, nis) for x in weighted_matrix])

        # Calculate Taxicab (Manhattan) distance
        T = np.sum(np.abs(weighted_matrix - nis), axis=1)
    
        # Construct the relative assessment matrix H
        h = np.zeros((m, m))
        for i, j in itertools.product(range(m), range(m)):
            h[i, j] = (E[i] - E[j]) + (self._psi(E[i] - E[j]) * (T[i] - T[j]))

        H = np.sum(h, axis=1)
        return H