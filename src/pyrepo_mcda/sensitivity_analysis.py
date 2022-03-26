import numpy as np
import pandas as pd
import copy

from .mcda_methods.codas import CODAS
from .mcda_methods.edas import EDAS
from .mcda_methods.mabac import MABAC
from .mcda_methods.multimoora import MULTIMOORA
from .mcda_methods.spotis import SPOTIS
from .mcda_methods.topsis import TOPSIS
from .mcda_methods.vikor import VIKOR
from .mcda_methods.waspas import WASPAS

from .additions import rank_preferences
from .normalizations import *
from .distance_metrics import *


class Sensitivity_analysis_weights():
    def __init__(self):
        """
        Create object of `Sensitivity_analysis_weights` method.
        """
        pass


    def __call__(self, matrix, weights, types, percentages, mcda_name, j):
        """
        Method for sensitivity analysis. This method determines rankings of alternatives using chosen
        MCDA method name `mcda_name` for different modifications of criterion `j` weight.

        Parameters
        -----------
            matrix : ndarray
                Decision matrix with alternatives performances data. This matrix includes
                data on m alternatives in rows considering criteria in columns

            weights : ndarray
                Vector with criteria weights. All weights in this vector must sum to 1.

            types : ndarray
                Vector with criteria types. Types can be equal to 1 for profit criteria and -1
                for cost criteria.

            percentages : ndarray
                Vector with percentage values of given criteria weight modification.

            mcda_name : str
                Name of applied MCDA method

            j : int
                Index of column in decision matrix `matrix` that indicates for which criterion
                the weight is modified.

        Returns
        --------
            data_sens : DataFrame
                dataframe with rankings calculated for subsequent modifications of criterion j weight
        """
        list_alt_names = [r'$A_{' + str(i) + '}$' for i in range(1, matrix.shape[0] + 1)]
        return Sensitivity_analysis_weights._sensitivity_analysis_weights(self, matrix, weights, types, percentages, mcda_name, list_alt_names, j)


    def _change_weights(self, j, weights, change_val):
        """
        Method for criteria weights modification in sensitivity analysis procedure.

        Parameters
        -----------
            j : int
                Index of column in decision matrix `matrix` that indicates for which criterion
                the weight is modified.

            weights : ndarray
                Vector of criteria weights

            change_val : float
                Percentage value of criterion weight modification

        Returns
        --------
            weights_copy : ndarray
                Vector with criteria weights after modification their values for sensitivity analysis
        """
        
        weights_copy = copy.deepcopy(weights)
        # Calculate value of selected criterion j weight modification
        change = weights_copy[j] * change_val
        # Calculate new value of selected criterion j weights
        new_weight = weights_copy[j] + change
        # Calculate new values of other criteria weights considering modification of j weights
        weights_copy = weights_copy - (change / (len(weights) - 1))
        # Assign new weight to criterion j
        weights_copy[j] = new_weight
        return weights_copy


    @staticmethod
    def _sensitivity_analysis_weights(self, matrix, weights, types, percentages, mcda_name, list_alt_names, j):
        # Create a dataframe for sensitivity analysis results (rankings)
        data_sens = pd.DataFrame()
        # Assisgn indexes (alternatives symbols) to dataframe `datasens`
        data_sens['Ai'] = list_alt_names
        # Iterate by two directions of weight modification: -1 for weight decreasing
        # and 1 for weight increasing
        for dir in [-1, 1]:
            # Sorting percentages in appropriate order for visualization results
            if dir == -1:
                direct_percentages = copy.deepcopy(percentages[::-1])
            else:
                direct_percentages = copy.deepcopy(percentages)
            # Iterate by values of weight change in vector `direct_percentages``
            for change_val in direct_percentages:
                # Change weights using method named `_change_weights` from class `Sensitivity_analysis_weights`
                weights_copy = self._change_weights(j, weights, dir * change_val)

                # Calculate alternatives ranking using selected MCDA method, `matrix``, vector of new weights
                # `weights_copy` and criteria types `types`
                if mcda_name == 'TOPSIS':
                    topsis = TOPSIS(normalization_method = minmax_normalization, distance_metric = euclidean)
                    pref = topsis(matrix, weights_copy, types)
                    rank = rank_preferences(pref, reverse = True)

                elif mcda_name == 'CODAS':
                    codas = CODAS(normalization_method = linear_normalization, distance_metric = euclidean, tau = 0.02)
                    pref = codas(matrix, weights_copy, types)
                    rank = rank_preferences(pref, reverse = True)

                elif mcda_name == 'VIKOR':
                    vikor = VIKOR(normalization_method = minmax_normalization)
                    pref = vikor(matrix, weights_copy, types)
                    rank = rank_preferences(pref, reverse = False)

                elif mcda_name == 'SPOTIS':
                    bounds_min = np.amin(matrix, axis = 0)
                    bounds_max = np.amax(matrix, axis = 0)
                    bounds = np.vstack((bounds_min, bounds_max))
                    spotis = SPOTIS()
                    pref = spotis(matrix, weights_copy, types, bounds)
                    rank = rank_preferences(pref, reverse = False)

                elif mcda_name == 'EDAS':
                    edas = EDAS()
                    pref = edas(matrix, weights_copy, types)
                    rank = rank_preferences(pref, reverse = True)

                elif mcda_name == 'MABAC':
                    mabac = MABAC(normalization_method = minmax_normalization)
                    pref = mabac(matrix, weights_copy, types)
                    rank = rank_preferences(pref, reverse = True)

                elif mcda_name == 'MULTIMOORA':
                    multimoora = MULTIMOORA()
                    pref_tab, rank = multimoora(matrix, weights_copy, types)

                elif mcda_name == 'WASPAS':
                    waspas = WASPAS(normalization_method = linear_normalization, lambda_param = 0.5)
                    pref = waspas(matrix, weights_copy, types)
                    rank = rank_preferences(pref, reverse = True)
                
                # Assign calculated ranking to column with value of j criterion weight change in `data_sens` dataframe
                data_sens['{:.0f}'.format(dir * change_val * 100) + '%'] = rank

        # Drop column with `-0%` name for visualization result
        if '-0%' in list(data_sens.columns):
            data_sens = data_sens.drop(['-0%'], axis = 1)
        # Set index with alternatives symbols in dataframe with sensitivity analysis results (rankings)
        data_sens = data_sens.set_index('Ai')
        return data_sens