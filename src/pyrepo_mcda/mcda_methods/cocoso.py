import numpy as np

from ..normalizations import minmax_normalization
from .mcda_method import MCDA_method


class COCOSO(MCDA_method):
    def __init__(self, normalization_method = minmax_normalization, lambda_param = 0.5):
        """
        Create the COCOSO method object and select value of lambda parameter called `lambda_param`.
        By default, the lambda parameter is equal to 0.5

        Parameters
        -----------
            lambda_param : parameter chosen by decision makers, usually is equal to 0.5. It determines
            flexibility and stability of the proposed CoCoSo
        """

        self.normalization_method = normalization_method
        self.lambda_param = lambda_param


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
        >>> cocoso = COCOSO(lambda_param = lambda_param)
        >>> pref = cocoso(matrix, weights, types)
        >>> rank = rank_preferences(pref, reverse = True)
        """

        COCOSO._verify_input_data(matrix, weights, types)
        return COCOSO._cocoso(matrix, weights, types, self.normalization_method, self.lambda_param)


    @staticmethod
    def _cocoso(matrix, weights, types, normalization_method = minmax_normalization, lambda_param = 0.5):
        # Normalize matrix using chosen normalization. minmax_normalization is default
        norm_matrix = normalization_method(matrix, types)
        S = np.sum(weights * norm_matrix, axis = 1)
        P = np.sum(norm_matrix**weights, axis = 1)

        kia = (P + S) / np.sum(P + S)
        kib = S / np.min(S) + P / np.min(P)
        kic = (lambda_param * S + (1 - lambda_param) * P) / (lambda_param * np.max(S) + (1 - lambda_param) * np.max(P))
        K = (kia * kib * kic)**(1/3) + 1/3*(kia + kib + kic)
        return K