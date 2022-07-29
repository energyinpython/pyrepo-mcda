import itertools
import numpy as np

from .promethee import PROMETHEE_II



# PROSA Examining Sustainability at the Criteria Level (PROSA-C)
class PROSA_C(PROMETHEE_II):
    def __init__(self):
        """
        Create the PROSA method object
        """
        pass


    def __call__(self, matrix, weights, types, preference_functions = None, p = None, q = None, s = None):
        PROSA_C._verify_input_data(matrix, weights, types)
        
        if preference_functions is None:
            preference_functions = [self._usual_function for pf in range(len(weights))]

        if p is None:
            u = np.sqrt(np.sum(np.square(np.mean(matrix, axis = 0) - matrix), axis = 0) / matrix.shape[0])
            p = 2 * u
            
        if q is None:
            u = np.sqrt(np.sum(np.square(np.mean(matrix, axis = 0) - matrix), axis = 0) / matrix.shape[0])
            q = 0.5 * u

        if s is None:
            s = np.repeat(0.3, len(weights))

        PROSA_C._verify_input_data(matrix, weights, types)
        # Verification of the correctness of given arguments
        if len(preference_functions) != np.shape(matrix)[1]:
            raise ValueError('The list of preference functions must be equal in length to the number of criteria')
        if len(p) != np.shape(matrix)[1]:
            raise ValueError('The length of the vector p must be equal to the number of criteria')
        if len(q) != np.shape(matrix)[1]:
            raise ValueError('The length of the vector q must be equal to the number of criteria')
        if len(s) != np.shape(matrix)[1]:
            raise ValueError('The length of the vector s must be equal to the number of criteria')

        return PROSA_C._prosa_c(self, matrix, weights, types, preference_functions, p, q, s)


    @staticmethod
    def _prosa_c(self, matrix, weights, types, preference_functions, p, q, s):
        """
        Score alternatives provided in the decision matrix `matrix` using criteria `weights` and criteria `types`.
        
        Parameters
        -----------
            matrix : ndarray
                Decision matrix with m alternatives in rows and n criteria in columns.
            weights: ndarray
                Criteria weights. The sum of weights must be equal to 1.
            types: ndarray
                Criteria types. Profit criteria are represented by 1 and cost by -1.
            preference_functions : list
                List with methods containing preference functions for calculating the
                preference degree for each criterion.
            p : ndarray
                Vector with values representing the threshold of absolute preference.
            q : ndarray
                Vector with values representing the threshold of indifference.
            s : ndarray
                Vector with values of the coefficient sj for the criteria
        
        Returns
        --------
            ndrarray
                Preference values of each alternative. The best alternative has the highest preference value. 
        
        Examples
        ----------
        >>> prosa_c = PROSA_C()
        >>> preference_functions = [prosa_c._linear_function for pf in range(len(weights))]
        >>> u = np.sqrt(np.sum(np.square(np.mean(matrix, axis = 0) - matrix), axis = 0) / matrix.shape[0])
        >>> p = 2 * u
        >>> q = 0.5 * u
        >>> s = np.repeat(0.3, len(weights))
        >>> pref = promethee_II(matrix, weights, types, preference_functions, p = p, q = q, s = s)
        >>> rank = rank_preferences(pref, reverse = True)
        """

        m, n = matrix.shape
        phi = np.zeros((m, n))

        # Determination of deviations based on pair-wise comparisons,
        # Application of the preference function,
        # Calculation of a single criterion net outranking flow
        # preference functions are inherited from the PROMETHEE II class
        for j, i, k in itertools.product(range(n), range(m), range(m)):
            phi[i, j] += preference_functions[j](types[j] * (matrix[i, j] - matrix[k, j]), p[j], q[j]) -\
                preference_functions[j](types[j] * (matrix[k, j] - matrix[i, j]), p[j], q[j])

        
        phi = phi / (m - 1)
        # Calculation of a global (overall) net outranking flow,
        # phi_net is the weighted sum of net flow for each criterion
        phi_net = np.sum(phi * weights, axis = 1)

        # Calculate the value of a mean absolute deviation in a weighted form,
        # where the sustainability (compensation) coefficient was taken into consideration,
        # where s denotes the sustainability (compensation) coefficient for a criterion j.
        WMAD = np.sum(np.abs(phi_net.reshape(-1, 1) - phi) * weights * s, axis = 1)

        # The final evaluation of alternatives (PSV_net) (PROSA net Sustainable Value), is calculated
        PSV_net = phi_net - WMAD

        return PSV_net