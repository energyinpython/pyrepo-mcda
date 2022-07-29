import numpy as np

from .mcda_method import MCDA_method


class MARCOS(MCDA_method):
    def __init__(self):
        """
        Create the MARCOS method object
        """
        pass

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
        >>> marcos = MARCOS()
        >>> pref = marcos(matrix, weights, types)
        >>> rank = rank_preferences(pref, reverse = True)
        """
        
        MARCOS._verify_input_data(matrix, weights, types)
        return MARCOS._marcos(matrix, weights, types)


    @staticmethod
    def _marcos(matrix, weights, types):
        # Step 1 - define the decision matrix with m alternatives and n criteria
        # a decision matrix is provided as an argument of the MARCOS method
        # Step 2
        # Create the ideal solution AI
        AI = np.zeros(matrix.shape[1])
        # for profit criteria
        AI[types == 1] = np.max(matrix[:, types == 1], axis = 0)
        # for cost criteria
        AI[types == -1] = np.min(matrix[:, types == -1], axis = 0)

        # Create the antiideal solution AAI
        AAI = np.zeros(matrix.shape[1])
        # for profit criteria
        AAI[types == 1] = np.min(matrix[:, types == 1], axis = 0)
        # for cost criteria
        AAI[types == -1] = np.max(matrix[:, types == -1], axis = 0)

        
        e_matrix = np.vstack((AAI, matrix))
        e_matrix = np.vstack((e_matrix, AI))

        # Step 3 normalization of extended matrix
        norm_matrix = np.zeros(e_matrix.shape)
        # for cost criteria
        norm_matrix[:, types == -1] = AI[types == -1] / e_matrix[:, types == -1]

        # for profit criteria
        norm_matrix[:, types == 1] = e_matrix[:, types == 1] / AI[types == 1]

        # Step 4 determination of the weighted matrix
        v = norm_matrix * weights

        # Step 5 Calculation of the utility degree of alternatives Ki
        Si = np.sum(v, axis = 1)
        Ki_minus = Si / np.sum(v[0, :])
        Ki_plus = Si / np.sum(v[-1, :])

        # Step 6 Determination of the utility function of alternatives f(Ki).
        f_Ki_minus = Ki_plus / (Ki_plus + Ki_minus)
        f_Ki_plus = Ki_minus / (Ki_plus + Ki_minus)
        f_Ki = (Ki_plus + Ki_minus) / (1 + ((1 - f_Ki_plus) / f_Ki_plus) + ((1 - f_Ki_minus) / f_Ki_minus))

        # Step 7 Ranking the alternatives. 
        # Ranking of the alternatives is based on the final values of utility functions. 
        # It is desirable that an alternative has the highest possible value of 
        # the utility function

        # return utility function values for each alternative
        return f_Ki[1:-1]