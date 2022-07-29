import numpy as np

from scipy.sparse.linalg import eigs

from .mcda_method import MCDA_method
from ..normalizations import minmax_normalization



class AHP(MCDA_method):
    def __init__(self, normalization_method = minmax_normalization):
        """
        Create the AHP method object

        Parameters
        -----------
            normalization_method : function
                If you use the AHP method to evaluate a matrix containing the numerical 
                values of the performance alternatives and you have a vector of criteria 
                weights containing their numerical values, then you are not using the 
                `classic_ahp` method but a method called `ahp`.
                You have to choose a method to normalize the given decision matrix.
                The default normalization technique is `minmax_normalization`
        """
        self.normalization_method = normalization_method


    def __call__(self, matrix, weights, types):
        """
        Score alternatives provided in decision matrix `matrix` using criteria `weights` and criteria `types`.

        Parameters
        ------------
        matrix : ndarray
                Decision matrix with numerical performance values of alternatives. Decision matrix 
                includes m alternatives in rows and n criteria in columns.
        weights: ndarray
            Vector with criteria weights given in numerical values. The sum of weights 
            must be equal to 1.
        types: ndarray
            Criteria types. Profit criteria are represented by 1 and cost by -1.

        Results
        ---------
            ndrarray
                Preference values of each alternative. The best alternative has the highest preference value. 

        Examples
        ----------
        >>> ahp = AHP()
        >>> pref = ahp(matrix, weights, types)
        >>> rank = rank_preferences(pref, reverse = True)
        """
        AHP._verify_input_data(matrix, weights, types)
        return AHP._ahp(self, matrix, weights, types, self.normalization_method)


    def _check_consistency(self, X):
        """
        Consistency Check on the Pairwise Comparison Matrix of the Criteria or alternatives

        Parameters
        -----------
            X : ndarray
                matrix of pairwise comparisons

        Examples
        ----------
        >>> PCcriteria = np.array([[1, 1, 5, 3], [1, 1, 5, 3], [1/5, 1/5, 1, 1/3], [1/3, 1/3, 3, 1]])
        >>> ahp = AHP()
        >>> ahp._check_consistency(PCcriteria)
        """
        n = X.shape[1]
        RI = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]
        lambdamax = np.amax(np.linalg.eigvals(X).real)
        CI = (lambdamax - n) / (n - 1)
        CR = CI / RI[n - 1]
        print("Inconsistency index: ", CR)
        if CR > 0.1:
            print("The pairwise comparison matrix is inconsistent")


    def _calculate_eigenvector(self, X):
        """
        Compute the Priority Vector of Criteria (weights) or alternatives using Eigenvector method

        Parameters
        -----------
            X : ndarray
                matrix of pairwise comparisons

        Returns
        ---------
            ndarray
                Eigenvector

        Examples
        ----------
        >>> PCM1 = np.array([[1, 5, 1, 1, 1/3, 3],
        [1/5, 1, 1/3, 1/5, 1/7, 1],
        [1, 3, 1, 1/3, 1/5, 1],
        [1, 5, 3, 1, 1/3, 3],
        [3, 7, 5, 3, 1, 7],
        [1/3, 1, 1, 1/3, 1/7, 1]])
        >>> ahp = AHP()
        >>> S = ahp._calculate_eigenvector(PCM1)
        """
        val, vec = eigs(X, k=1)
        eig_vec = np.real(vec)
        S = eig_vec / np.sum(eig_vec)
        S = S.ravel()
        return S


    def _normalized_column_sum(self, X):
        """
        Compute the Priority Vector of Criteria (weights) or alternatives using The normalized column sum method

        Parameters
        -----------
            X : ndarray
                matrix of pairwise comparisons

        Returns
        ---------
            ndarray
                Vector with weights calculated with The normalized column sum method

        Examples
        ----------
        >>> PCM1 = np.array([[1, 5, 1, 1, 1/3, 3],
        [1/5, 1, 1/3, 1/5, 1/7, 1],
        [1, 3, 1, 1/3, 1/5, 1],
        [1, 5, 3, 1, 1/3, 3],
        [3, 7, 5, 3, 1, 7],
        [1/3, 1, 1, 1/3, 1/7, 1]])
        >>> ahp = AHP()
        >>> S = ahp._normalized_column_sum(PCM1)
        """
        return np.sum(X, axis = 1) / np.sum(X)


    def _geometric_mean(self, X):
        """
        Compute the Priority Vector of Criteria (weights) or alternatives using The geometric mean method

        Parameters
        -----------
            X : ndarray
                matrix of pairwise comparisons

        Returns
        ---------
            ndarray
                Vector with weights calculated with The geometric mean method

        Examples
        ----------
        >>> PCM1 = np.array([[1, 5, 1, 1, 1/3, 3],
        [1/5, 1, 1/3, 1/5, 1/7, 1],
        [1, 3, 1, 1/3, 1/5, 1],
        [1, 5, 3, 1, 1/3, 3],
        [3, 7, 5, 3, 1, 7],
        [1/3, 1, 1, 1/3, 1/7, 1]])
        >>> ahp = AHP()
        >>> S = ahp._geometric_mean(PCM1)
        """
        n = X.shape[1]
        numerator = (np.prod(X, axis = 1))**(1 / n)
        denominator = np.sum(numerator)
        return numerator / denominator

    
    def _classic_ahp(self, alt_matrices, weights, calculate_priority_vector_method = None):
        """
        Calculate the global alternative priorities.
        This is a method for classic AHP where you provide matrices with values of pairwise 
        comparisons of alternatives and weights in the form of a priority vector.

        Parameters
        ------------
            alt_matrices : list
                list with matrices including values of pairwise comparisons of alternatives
            weights : ndarray
                priority vector of criteria (weights)
            calculate_priority_vector_method : function
                Method for calculation of the priority vector. It can be chosen from three
                available methods: _calculate_eigenvector, _normalized_column_sum and
                _geometric_mean
                if the user does not provide calculate_priority_vector_method, it is automatically
                set as the default _calculate_eigenvector

        Returns
        ---------
            ndarray
                vector with the global alternative priorities

        Examples
        -----------
        >>> PCcriteria = np.array([[1, 1, 5, 3], [1, 1, 5, 3], 
        [1/5, 1/5, 1, 1/3], [1/3, 1/3, 3, 1]])
        >>> PCM1 = np.array([[1, 5, 1, 1, 1/3, 3],
        [1/5, 1, 1/3, 1/5, 1/7, 1],
        [1, 3, 1, 1/3, 1/5, 1],
        [1, 5, 3, 1, 1/3, 3],
        [3, 7, 5, 3, 1, 7],
        [1/3, 1, 1, 1/3, 1/7, 1]])
        >>> PCM2 = np.array([[1, 7, 3, 1/3, 1/3, 1/3],
        [1/7, 1, 1/3, 1/7, 1/9, 1/7],
        [1/3, 3, 1, 1/5, 1/5, 1/5],
        [3, 7, 5, 1, 1, 1],
        [3, 9, 5, 1, 1, 1],
        [3, 7, 5, 1, 1, 1]])
        >>> PCM3 = np.array([[1, 1/9, 1/7, 1/9, 1, 1/5],
        [9, 1, 1, 1, 5, 3],
        [7, 1, 1, 1, 5, 1],
        [9, 1, 1, 1, 7, 3],
        [1, 1/5, 1/5, 1/7, 1, 1/3],
        [5, 1/3, 1, 1/3, 3, 1]])
        >>> PCM4 = np.array([[1, 1/5, 1/5, 1/3, 1/7, 1/5],
        [5, 1, 1, 3, 1/3, 1],
        [5, 1, 1, 1, 1/3, 1],
        [3, 1/3, 1, 1, 1/7, 1],
        [7, 3, 3, 7, 1, 5],
        [5, 1, 1, 1, 1/5, 1]])

        >>> ahp = AHP()
        >>> ahp._check_consistency(PCcriteria)
        >>> weights = ahp._calculate_eigenvector(PCcriteria)
        >>> alt_matrices = []
        >>> alt_matrices.append(PCM1)
        >>> alt_matrices.append(PCM2)
        >>> alt_matrices.append(PCM3)
        >>> alt_matrices.append(PCM4)

        >>> calculate_priority_vector_method = ahp._calculate_eigenvector
        >>> pref = ahp._classic_ahp(alt_matrices, weights, calculate_priority_vector_method)
        >>> rank = rank_preferences(pref, reverse = True)
        """

        # eigenvector method is default method to calculate priority vector
        if calculate_priority_vector_method is None:
            calculate_priority_vector_method = self._calculate_eigenvector
        # Check consistency of all pairwise comparison matrices for alternatives 
        for alt in alt_matrices:
            self._check_consistency(alt)

        m = alt_matrices[0].shape[0]
        n = len(weights)

        # Calculate priority vector withe selected method
        S = np.zeros((m, n))
        for el, alt in enumerate(alt_matrices):
            S[:, el] = calculate_priority_vector_method(alt)

        # Calculate the global alternative priorities
        # Calculate the weighted matrix
        Sw = S * weights
        # Aggregate the Local Priorities and Rank the Alternatives
        S_final = np.sum(Sw, axis = 1) / np.sum(Sw)

        return S_final


    @staticmethod
    def _ahp(self, matrix, weights, types, normalization_method):
        """
        Score alternatives provided in decision matrix `matrix` using criteria `weights` and criteria `types`.

        Parameters
        ------------
        matrix : ndarray
                Decision matrix with numerical performance values of alternatives. The decision matrix 
                includes m alternatives in rows and n criteria in columns.
        weights: ndarray
            Vector with criteria weights given in numerical values. The sum of weights 
            must be equal to 1.
        types: ndarray
            Criteria types. Profit criteria are represented by 1 and cost by -1.

        Results
        ---------
            ndrarray
                Preference values of each alternative. The best alternative has the highest preference value. 

        Examples
        ----------
        >>> ahp = AHP()
        >>> pref = ahp(matrix, weights, types)
        >>> rank = rank_preferences(pref, reverse = True)
        """
        nmatrix = normalization_method(matrix, types)
        weighted_matrix = nmatrix * weights
        pref = np.sum(weighted_matrix, axis = 1)
        return pref