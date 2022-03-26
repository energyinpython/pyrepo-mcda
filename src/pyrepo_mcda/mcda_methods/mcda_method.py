from abc import ABC
import numpy as np

class MCDA_method(ABC):

    def __call__(self, matrix, weights, types):
        """Score alternatives from decision matrix `matrix` using criteria weights `weights` and criteria types `types`

        Parameters
        ----------
            matrix : ndarray
                decision matrix with performance values for m alternatives in rows and n criteria in columns
            weights : ndarray
                vector with criteria weights with size equal to number of columns n of `matrix`
            types : ndarray
                vector with criteria types containing values of 1 for profit criteria and -1 for cost criteria
                with size equal to number of columns n of `matrix`
        """
        pass

    @staticmethod
    def _verify_input_data(matrix, weights, types):
        m, n = matrix.shape
        if len(weights) != n:
            raise ValueError('The size of the weight vector must be the same as the number of criteria')
        if len(types) != n:
            raise ValueError('The size of the types vector must be the same as the number of criteria')
        check_types = np.all((types == 1) | (types == -1))
        if check_types == False:
            raise ValueError('Criteria types can only have a value of 1 for profits and -1 for costs')