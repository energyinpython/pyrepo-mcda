import numpy as np
from .mcda_method import MCDA_method

class VMCM(MCDA_method):
    def __init__(self):
        """
        Create the VMCM method object.
        """

        pass


    def _elimination(self, matrix):
        """
        Calculate significance coefficient values for each criterion.
        Criteria with significance coefficient values between 0 and 0.1 are recommended to be eliminated from the considered criteria set.

        Parameters
        --------------
            matrix : ndarray
                Decision matrix with m alternatives in rows and n criteria in columns.

        Examples
        --------------
        >>> vmcm = VMCM()
        >>> vmcm._elimination(matrix)
        """

        # Elimination of variables
        v = np.std(matrix, axis = 0, ddof = 1) / np.mean(matrix, axis = 0)
        to_eliminate = []
        print('Elimination of variables stage (significance coefficient of features):')
        for el, vj in enumerate(v):
            print(f'C{el + 1} = {vj:.4f}')
            if 0 <= vj <= 0.1:
                to_eliminate.append('C' + str(el + 1))
        print('Criteria to eliminate:')
        if not(to_eliminate):
            print('None')
        for te in to_eliminate:
            print(te)



    def _weighting(self, matrix):
        """
        Calculate criteria weights

        Parameters
        -------------
            matrix : ndarray
                Decision matrix with m alternatives in rows and n criteria in columns.

        Returns
        -------------
            ndarray
                Vector with criteria weights

        Examples
        ------------
        >>> vmcm = VMCM()
        >>> weights = vmcm._weighting(matrix)
        """

        v = np.std(matrix, axis = 0, ddof = 1) / np.mean(matrix, axis = 0)
        return v / np.sum(v)
    

    def _pattern_determination(self, matrix, types):
        """
        Automatic determination of pattern and anti-pattern

        Parameters
        --------------
            matrix : ndarray
                Decision matrix with m alternatives in rows and n criteria in columns.

        Returns
        --------------
            ndarray, ndarray
                Two vectors including values respectively of pattern and anti-pattern

        Examples
        --------------
        >>> vmcm = VMCM()
        >>> pattern, antipattern = vmcm._pattern_determination(matrix, types)
        """

        # Normalization of variables
        norm_matrix = (matrix - np.mean(matrix, axis = 0)) / np.std(matrix, axis = 0, ddof = 1)

        # Determination of pattern and anti-pattern
        q1 = np.quantile(norm_matrix, 0.25, axis = 0)
        q3 = np.quantile(norm_matrix, 0.75, axis = 0)

        pattern = np.zeros(matrix.shape[1])
        anti_pattern = np.zeros(matrix.shape[1])

        pattern[types == 1] = q3[types == 1]
        pattern[types == -1] = q1[types == -1]

        anti_pattern[types == 1] = q1[types == 1]
        anti_pattern[types == -1] = q3[types == -1]

        return pattern, anti_pattern
    

    def _classification(self, m):
        """
        Assign evaluated objects to classes

        Parameters
        -------------
            m : ndarray
                Vector with values of synthetic measure

        Returns
        -------------
            ndarray
                Vector including classes assigned to evaluated objects

        Examples
        --------------
        >>> vmcm = VMCM()
        >>> pref = vmcm(matrix, weights, types)
        >>> classes = vmcm._classification(pref)
        """

        # Classification of objects
        m_mean = np.mean(m)
        m_std = np.std(m, ddof = 1)

        cl = np.zeros(len(m))

        cl[m >= m_mean + m_std] = 1
        cl[np.logical_and(m >= m_mean, m < m_mean + m_std)] = 2
        cl[np.logical_and(m >= m_mean - m_std, m < m_mean)] = 3
        cl[m < m_mean - m_std] = 4

        return cl


    def __call__(self, matrix, weights, types, pattern, anti_pattern):
        """
        Score alternatives provided in decision matrix `matrix` with m alternatives in rows and 
        n criteria in columns using criteria `weights` and criteria `types`.

        Parameters
        ----------
            matrix : ndarray
                Decision matrix with m alternatives in rows and n criteria in columns.
            weights: ndarray
                Vector with criteria weights. Sum of weights must be equal to 1.
            types : ndarray
                Vector with criteria types. Profit criteria are represented by 1 and cost by -1.
            pattern : ndarray
                Vector with values of pattern
            anti_pattern : ndarray
                Vector with values of anti-pattern

        Returns
        -------
            ndrarray
                Vector with preference values of each alternative. The best alternative has the highest preference value. 

        Examples
        ---------
        >>> vmcm = VMCM()
        >>> pattern, antipattern = vmcm._pattern_determination(matrix, types)
        >>> pref = vmcm(matrix, weights, types, pattern, antipattern)
        >>> rank = rank_preferences(pref, reverse = True)
        """

        VMCM._verify_input_data(matrix, weights, types)
        return VMCM._vmcm(matrix, weights, types, pattern, anti_pattern)


    @staticmethod
    def _vmcm(matrix, weights, types, pattern, anti_pattern):
        # Normalization of variables
        norm_matrix = (matrix - np.mean(matrix, axis = 0)) / np.std(matrix, axis = 0, ddof = 1)

        # Construction of the synthetic measure
        m = np.sum((norm_matrix - anti_pattern) * (pattern - anti_pattern), axis = 1) / np.sum((pattern - anti_pattern)**2)

        return m