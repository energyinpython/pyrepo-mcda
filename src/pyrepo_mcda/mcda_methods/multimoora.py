import numpy as np

from ..normalizations import multimoora_normalization
from ..additions import rank_preferences
from ..compromise_rankings import dominance_directed_graph
from .mcda_method import MCDA_method


class MULTIMOORA_RS(MCDA_method):
    def __init__(self):
        """Create object of the MULTIMOORA Ratio System (RS) method. This method is an integral part of the 
        MULTIMOORA method. This method is the same as the MOORA method, and it can be used 
        independently as a separate MCDA method"""
        pass


    def __call__(self, matrix, weights, types):
        """
        Score alternatives provided in decision matrix `matrix` using vector with criteria weights
        `weights` and vector with criteria types `types`.

        Parameters
        ----------
            matrix : ndarray
                Decision matrix with m alternatives in rows and n criteria in columns.
            weights: ndarray
                Criteria weights. Sum of weights must be equal to 1.
            types: ndarray
                Criteria types. Profit criteria are represented by 1 and cost by -1.

        Returns
        -------
            ndrarray
                Preference values of each alternative. The best alternative has the highest preference value. 

        Examples
        ---------
        >>> multimoora_rs = MULTIMOORA_RS()
        >>> pref = multimoora_rs(matrix, weights, types)
        >>> rank = rank_preferences(pref, reverse = True)
        """
        MULTIMOORA_RS._verify_input_data(matrix, weights, types)
        return MULTIMOORA_RS._multimoora_rs(matrix, weights, types)


    @staticmethod
    def _multimoora_rs(matrix, weights, types):
        # Normalize decision matrix using normalization method `multimoora_normalization` dedicated to the MULTIMOORA method
        norm_matrix = multimoora_normalization(matrix)
        # Calculate the overall performance index of alternatives as 
        # difference between sums of weighted normalized performance ratings
        # of the profit and cost criteria 
        Q_sum_profit = np.sum(weights[types == 1] * norm_matrix[:, types == 1], axis = 1)
        Q_sum_cost = np.sum(weights[types == -1] * norm_matrix[:, types == -1], axis = 1)
        Q = Q_sum_profit - Q_sum_cost
        return Q


class MULTIMOORA_RP(MCDA_method):
    def __init__(self):
        """Create object of the MULTIMOORA Reference Point (RP) method. This method is an integral part of the MULTIMOORA
        method"""
        pass


    def __call__(self, matrix, weights, types):
        """
        Score alternatives provided in decision matrix `matrix` using vector with criteria weights
        `weights` and vector with criteria types `types`.

        Parameters
        ----------
            matrix : ndarray
                Decision matrix with m alternatives in rows and n criteria in columns.
            weights: ndarray
                Criteria weights. Sum of weights must be equal to 1.
            types: ndarray
                Criteria types. Profit criteria are represented by 1 and cost by -1.

        Returns
        -------
            ndrarray
                Preference values of each alternative. The best alternative has the lowest preference value. 

        Examples
        ---------
        >>> multimoora_rp = MULTIMOORA_RP()
        >>> pref = multimoora_rp(matrix, weights, types)
        >>> rank = rank_preferences(pref, reverse = False)
        """
        MULTIMOORA_RP._verify_input_data(matrix, weights, types)
        return MULTIMOORA_RP._multimoora_rp(matrix, weights, types)


    @staticmethod
    def _multimoora_rp(matrix, weights, types):
        # Normalize decision matrix using normalization method `multimoora_normalization` dedicated to the MULTIMOORA method
        norm_matrix = multimoora_normalization(matrix)
    
        RR = np.zeros(norm_matrix.shape[1])

        maximums = np.amax(norm_matrix, axis = 0)
        minimums = np.amin(norm_matrix, axis = 0)
        RR[types == 1] = maximums[types == 1]
        RR[types == -1] = minimums[types == -1]
        weighted_matrix = weights * np.abs(RR - norm_matrix)
        A = np.max(weighted_matrix, axis = 1)
        return A


class MULTIMOORA_FMF(MCDA_method):
    def __init__(self):
        """Create object of the MULTIMOORA Full Multiplicative Form (FMF) method. This method is an integral part of the MULTIMOORA
        method"""
        pass


    def __call__(self, matrix, weights, types):
        """
        Score alternatives provided in decision matrix `matrix` using vector with criteria weights
        `weights` and vector with criteria types `types`.

        Parameters
        ----------
            matrix : ndarray
                Decision matrix with m alternatives in rows and n criteria in columns.
            weights: ndarray
                Criteria weights. Sum of weights must be equal to 1.
            types: ndarray
                Criteria types. Profit criteria are represented by 1 and cost by -1.

        Returns
        -------
            ndrarray
                Preference values of each alternative. The best alternative has the highest preference value. 

        Examples
        ---------
        >>> multimoora_fmf = MULTIMOORA_FMF()
        >>> pref = multimoora_fmf(matrix, weights, types)
        >>> rank = rank_preferences(pref, reverse = True)
        """
        MULTIMOORA_FMF._verify_input_data(matrix, weights, types)
        return MULTIMOORA_FMF._multimoora_fmf(matrix, weights, types)


    @staticmethod
    def _multimoora_fmf(matrix, weights, types):
        # Normalize decision matrix using normalization method `multimoora_normalization` dedicated to the MULTIMOORA method
        norm_matrix = multimoora_normalization(matrix)
        A = np.prod(weights[types == 1] * norm_matrix[:, types == 1], axis = 1)
        B = np.prod(weights[types == -1] * norm_matrix[:, types == -1], axis = 1)
        U = A / B
        return U


class MULTIMOORA(MCDA_method):
    def __init__(self, compromise_rank_method = dominance_directed_graph):
        """Create object of the MULTIMOORA method.

        Parameters:
        -----------
            compromise_rank_method : function
                method determining compromise ranking based on RS, RP and FMF MULTIMOORA rankings.
                The compromise rank method is selected from `compromise_rankings`.
                It can be `borda_copeland_compromise_ranking`, `dominance_directed_graph`,
                `rank_position_method` and `improved_borda_rule` which is dedicated to the MULTIMOORA method
            
        """
        self.compromise_rank_method = compromise_rank_method


    def __call__(self, matrix, weights, types):
        """
        Score alternatives provided in decision matrix `matrix` using vector with criteria weights
        `weights` and vector with criteria types `types`.

        Parameters
        ----------
            matrix : ndarray
                Decision matrix with m alternatives in rows and n criteria in columns.
            weights: ndarray
                Criteria weights. Sum of weights must be equal to 1.
            types: ndarray
                Criteria types. Profit criteria are represented by 1 and cost by -1.

        Returns
        -------
            ndrarray
                Preference values of each alternative. The best alternative has the highest preference value. 

        Examples
        ----------
        >>> multimoora = MULTIMOORA()
        >>> rank = multimoora(matrix, weights, types)
        """
        MULTIMOORA._verify_input_data(matrix, weights, types)
        return MULTIMOORA._multimoora(matrix, weights, types, self.compromise_rank_method)


    def _multimoora(matrix, weights, types, compromise_rank_method):

        # Calculate preference values and ranking of alternatives by MULTIMOORA RS
        multimoora_rs = MULTIMOORA_RS()
        pref_rs = multimoora_rs(matrix, weights, types)

        # in MULTIMOORA RS alternatives are sorted in descending order like in TOPSIS
        rank_rs = rank_preferences(pref_rs, reverse = True)

        # Calculate preference values and ranking of alternatives by MULTIMOORA RP
        multimoora_rp = MULTIMOORA_RP()
        pref_rp = multimoora_rp(matrix, weights, types)
        pref = np.hstack((pref_rs.reshape(-1,1), pref_rp.reshape(-1,1)))
        
        # in MULTIMOORA RP alternatives are sorted in ascending order like in VIKOR
        rank_rp = rank_preferences(pref_rp, reverse = False)
        rank = np.hstack((rank_rs.reshape(-1,1), rank_rp.reshape(-1,1)))

        # Calculate preference values and ranking of alternatives by MULTIMOORA FMF
        multimoora_fmf = MULTIMOORA_FMF()
        pref_fmf = multimoora_fmf(matrix, weights, types)
        pref = np.hstack((pref, pref_fmf.reshape(-1,1)))
        
        # in MULTIMOORA FMF alternatives are sorted in descending order like in TOPSIS
        rank_fmf = rank_preferences(pref_fmf, reverse = True)
        rank = np.hstack((rank, rank_fmf.reshape(-1,1)))

        if compromise_rank_method.__name__ == 'improved_borda_rule':
            mmoora_rank = compromise_rank_method(pref, rank)
        else:
            mmoora_rank = compromise_rank_method(rank)
            
        return mmoora_rank