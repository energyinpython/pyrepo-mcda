import numpy as np
from .mcda_method import MCDA_method


class EDAS(MCDA_method):
    def __init__(self):
        """Create object of the EDAS method"""
        pass


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
                Vevtor with criteria types. Profit criteria are represented by 1 and cost by -1.

        Returns
        --------
            ndrarray
                Vector with preference values of each alternative. The best alternative has the highest preference value. 

        Examples
        ---------
        >>> edas = EDAS()
        >>> pref = edas(matrix, weights, types)
        >>> rank = rank_preferences(pref, reverse = True)
        """
        EDAS._verify_input_data(matrix, weights, types)
        return EDAS._edas(matrix, weights, types)


    def _edas(matrix, weights, types):
        m, n = matrix.shape
        #AV = np.mean(matrix, axis = 0)

        # Calculate the average solution for each criterion
        AV = np.sum(matrix, axis = 0) / m

        # Calculate the Positive Distance (PDA) and Negative Distance (NDA) from average solution
        PDA = np.zeros(matrix.shape)
        NDA = np.zeros(matrix.shape)

        for j in range(0, n):
            if types[j] == 1:
                PDA[:, j] = (matrix[:, j] - AV[j]) / AV[j]
                NDA[:, j] = (AV[j] - matrix[:, j]) / AV[j]
            else:
                PDA[:, j] = (AV[j] - matrix[:, j]) / AV[j]
                NDA[:, j] = (matrix[:, j] - AV[j]) / AV[j]

        PDA[PDA < 0] = 0
        NDA[NDA < 0] = 0

        # Calculate the weighted sum of PDA and NDA for all alternatives
        SP = np.sum(weights * PDA, axis = 1)
        SN = np.sum(weights * NDA, axis = 1)

        # Normalize obtained values
        NSP = SP / np.max(SP)
        NSN = 1 - (SN / np.max(SN))

        # Calculate the appraisal score (AS) for each alternative 
        AS = (NSP + NSN) / 2
        return AS

