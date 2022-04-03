import numpy as np
from .mcda_method import MCDA_method

class SPOTIS(MCDA_method):
    def __init__(self):
        """Create SPOTIS method object.
        """
        pass


    def __call__(self, matrix, weights, types, bounds):
        """Score alternatives provided in decision matrix `matrix` using criteria `weights` and criteria `types`.

        Parameters
        -----------
            matrix : ndarray
                Decision matrix with m alternatives in rows and n criteria in columns.
            weights: ndarray
                Vector with criteria weights. Sum of weights must be equal to 1.
            types: ndarray
                Vector with criteria types. Profit criteria are represented by 1 and cost by -1.
            bounds: ndarray
                Bounds is ndarray with 2 rows and number of columns equal to criteria number. 
                Bounds contain minimum values in the first row and maximum values in the second row 
                for each criterion. Minimum and maximum values for the same criterion cannot be 
                the same.

        Returns
        --------
            ndrarray
                Vector with preference values of each alternative. The best alternative has the lowest preference value. 

        Examples
        ----------
        >>> bounds_min = np.amin(matrix, axis = 0)
        >>> bounds_max = np.amax(matrix, axis = 0)
        >>> bounds = np.vstack((bounds_min, bounds_max))
        >>> spotis = SPOTIS()
        >>> pref = spotis(matrix, weights, types, bounds)
        >>> rank = rank_preferences(pref, reverse = False)
        """
        SPOTIS._verify_input_data(matrix, weights, types)
        return SPOTIS._spotis(matrix, weights, types, bounds)


    @staticmethod
    def _spotis(matrix, weights, types, bounds):
        # Determine Ideal Solution Point (ISP)
        isp = np.zeros(matrix.shape[1])
        isp[types == 1] = bounds[1, types == 1]
        isp[types == -1] = bounds[0, types == -1]

        # Calculate normalized distances
        norm_matrix = np.abs(matrix - isp) / np.abs(bounds[1, :] - bounds[0, :])
        # Calculate the normalized weighted average distance
        D = np.sum(weights * norm_matrix, axis = 1)
        return D
