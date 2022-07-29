import numpy as np

from .mcda_method import MCDA_method
from ..normalizations import linear_normalization


class CRADIS(MCDA_method):
    def __init__(self, normalization_method = linear_normalization):
        """
        Create the CRADIS method object
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
        >>> cradis = CRADIS()
        >>> pref = cradis(matrix, weights, types)
        >>> rank = rank_preferences(pref, reverse = True)
        """
        CRADIS._verify_input_data(matrix, weights, types)
        return CRADIS._cradis(matrix, weights, types, self.normalization_method)


    @staticmethod
    def _cradis(matrix, weights, types, normalization_method):

        # Normalize the decision matrix
        norm_matrix = normalization_method(matrix, types)
        # Create the weighted normalized decision matrix
        v = norm_matrix * weights

        # Determine the ideal and anti-ideal solution

        # calculation of the ideal solution is done by finding the largest value in 
        # weighted normalized matrix v(max)
        # calculation of the anti-ideal solution is done by finding the smallest value in
        # weighted normalized matrix v(min)

        # Calculation of deviations from ideal and anti-ideal solutions
        # Calculating the grades of the deviation of individual alternatives from ideal and 
        # anti-ideal solutions
        Sp = np.sum(np.max(v) - v, axis = 1)
        Sm = np.sum(v - np.min(v), axis = 1)

        # Calculation of the utility function for each alternative in relation to the deviations 
        # from the optimal alternatives

        # Sop is the optimal alternative that has the smallest distance from the ideal solution
        # Som is the optimal alternative that has the greatest distance from the anti-ideal solution

        Sop = np.sum(np.max(v) - np.max(v, axis = 0))
        Som = np.sum(np.max(v, axis = 0) - np.min(v))

        Kp = Sop / Sp
        Km = Sm / Som
        Q = (Kp + Km) / 2

        # The best alternative is the one that has the highest value
        return Q