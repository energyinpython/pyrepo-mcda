import numpy as np

from .mcda_method import MCDA_method


class VIKOR(MCDA_method):
    """
    VIseKriterijumska Optimizacija I Kompromisno Resenje (VIKOR)
    method for ranking alternatives and identifying compromise solutions
    based on group utility and individual regret measures.
    """
    
    def __init__(self, normalization_method = None, v = 0.5):
        """Create the VIKOR method object.

        Parameters
        -----------
            normalization_method : function
                VIKOR does not use normalization by default, thus `normalization_method` is set to None by default.
                However, you can choose method for normalization of decision matrix chosen `normalization_method` from `normalizations`.
                It is used in a way `normalization_method(X, types)` where `X` is a decision matrix
                and `types` is a vector with criteria types where 1 means profit and -1 means cost.

            v : float
                parameter that is weight of strategy of the majority of criteria (the maximum group utility)
        """
        if not 0 <= v <= 1:
            raise ValueError("Parameter v must be between 0 and 1.")
        self.v = v
        self.normalization_method = normalization_method


    def __call__(self, matrix, weights, types):
        """
        Score alternatives provided in decision matrix `matrix` using criteria `weights` and criteria `types`.

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
                Vector with preference values of each alternative. The best alternative has the lowest preference value.

        Examples
        ---------
        >>> vikor = VIKOR(normalization_method = minmax_normalization)
        >>> pref = vikor(matrix, weights, types)
        >>> rank = rank_preferences(pref, reverse = False) 
        """
        VIKOR._verify_input_data(matrix, weights, types)
        return VIKOR._vikor(matrix, weights, types, self.normalization_method, self.v)


    @staticmethod
    def _vikor(matrix, weights, types, normalization_method, v):
        # Without special normalization method
        if normalization_method is None:

            # Determine the best `fstar` and the worst `fmin` values of all criterion function
            maximums_matrix = np.amax(matrix, axis = 0)
            minimums_matrix = np.amin(matrix, axis = 0)

            fstar = np.zeros(matrix.shape[1])
            fmin = np.zeros(matrix.shape[1])

            # for profit criteria (`types` == 1) and for cost criteria (`types` == -1)
            fstar[types == 1] = maximums_matrix[types == 1]
            fstar[types == -1] = minimums_matrix[types == -1]
            fmin[types == 1] = minimums_matrix[types == 1]
            fmin[types == -1] = maximums_matrix[types == -1]

            data_matrix = matrix

        else:
            # With special normalization method
            data_matrix = normalization_method(matrix, types)

            fstar = np.amax(data_matrix, axis=0)
            fmin = np.amin(data_matrix, axis=0)

        # Calculate the weighted matrix
        mask = (fstar != fmin)

        weighted_matrix = np.zeros_like(data_matrix, dtype=float)

        weighted_matrix[:, mask] = (
            weights[mask] *
            (fstar[mask] - data_matrix[:, mask]) /
            (fstar[mask] - fmin[mask])
        )

        # Calculate the `S` and `R` values
        S = np.sum(weighted_matrix, axis = 1)
        R = np.amax(weighted_matrix, axis = 1)
        # Calculate the Q values
        Sstar = np.min(S)
        Smin = np.max(S)

        if Sstar == Smin:
            S_normalized = np.zeros_like(S)
        else:
            S_normalized = (S - Sstar) / (Smin - Sstar)

        Rstar = np.min(R)
        Rmin = np.max(R)

        if Rstar == Rmin:
            R_normalized = np.zeros_like(R)
        else:
            R_normalized = (R - Rstar) / (Rmin - Rstar)

        Q = v * S_normalized + (1 - v) * R_normalized
        return Q
