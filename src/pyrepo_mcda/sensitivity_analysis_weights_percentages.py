import copy
import numpy as np
import pandas as pd

from .additions import rank_preferences


class Sensitivity_analysis_weights_percentages():
    def __init__(self):
        """
        Create object of `Sensitivity_analysis_weights_percentages` method.
        """
        pass


    def __call__(self, matrix, weights, types, percentages, method, j, dir_list):
        """
        Method for sensitivity analysis. This method determines rankings of alternatives using chosen
        MCDA method name `mcda_name` for different modifications of criterion `j` weight.

        Parameters
        -----------
            matrix : ndarray
                Decision matrix with alternatives performances data. This matrix includes
                values of alternatives performances in rows considering criteria in columns

            weights : ndarray
                Vector with criteria weights. All weights in this vector must sum to 1.

            types : ndarray
                Vector with criteria types. Types can be equal to 1 for profit criteria and -1
                for cost criteria.

            percentages : ndarray
                Vector with percentage values of given criteria weight modification in range from 0 to 1.

            method : class
                Initialized object of class of chosen MCDA method

            j : int
                Index of column in decision matrix `matrix` that indicates for which criterion
                the weight is modified.

            dir_list : list
                list with directions (signs of value) of criterion weight modification. 1 denotes increasing,
                and -1 denotes decreasing weight value. You can provide [-1, 1] for increasing and
                decreasing, [-1] for only decreasing, or [1] for only increasing chosen criterion weight.

        Returns
        --------
            data_sens : DataFrame
                dataframe with rankings calculated for subsequent modifications of criterion j weight

        Examples
        ----------
        >>> sensitivity_analysis = Sensitivity_analysis_weights_percentages()
        >>> df_sens = sensitivity_analysis(matrix, weights, types, percentages, method, j, [-1, 1])
        """
        list_alt_names = [r'$A_{' + str(i) + '}$' for i in range(1, matrix.shape[0] + 1)]
        return Sensitivity_analysis_weights_percentages._sensitivity_analysis_weights_percentages(self, matrix, weights, types, percentages, method, list_alt_names, j, dir_list)


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
                Percentage value of criterion weight modification in range from 0 to 1

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
    def _sensitivity_analysis_weights_percentages(self, matrix, weights, types, percentages, method, list_alt_names, j, dir_list):
        # Create a dataframe for sensitivity analysis results (rankings)
        data_sens = pd.DataFrame()
        # Assisgn indexes (alternatives symbols) to dataframe `data_sens`
        data_sens['Ai'] = list_alt_names
        # Iterate by two directions of weight modification: -1 for weight decreasing
        # and 1 for weight increasing
        for dir in dir_list:
            # Sorting percentages in appropriate order for visualization results
            if dir == -1:
                direct_percentages = copy.deepcopy(percentages[::-1])
            else:
                direct_percentages = copy.deepcopy(percentages)
            # Iterate by values of weight change in vector `direct_percentages`
            for change_val in direct_percentages:
                # Change weights using method named `_change_weights` from class `Sensitivity_analysis_weights_percentages`
                weights_copy = self._change_weights(j, weights, dir * change_val)

                # Calculate alternatives ranking using selected MCDA method, `matrix`, vector of new weights
                # `weights_copy` and criteria types `types`
                if method.__class__.__name__ == 'TOPSIS':
                    pref = method(matrix, weights_copy, types)
                    rank = rank_preferences(pref, reverse = True)

                elif method.__class__.__name__ == 'CODAS':
                    pref = method(matrix, weights_copy, types)
                    rank = rank_preferences(pref, reverse = True)

                elif method.__class__.__name__ == 'VIKOR':
                    pref = method(matrix, weights_copy, types)
                    rank = rank_preferences(pref, reverse = False)

                elif method.__class__.__name__ == 'SPOTIS':
                    bounds_min = np.amin(matrix, axis = 0)
                    bounds_max = np.amax(matrix, axis = 0)
                    bounds = np.vstack((bounds_min, bounds_max))
                    pref = method(matrix, weights_copy, types, bounds)
                    rank = rank_preferences(pref, reverse = False)

                elif method.__class__.__name__ == 'EDAS':
                    pref = method(matrix, weights_copy, types)
                    rank = rank_preferences(pref, reverse = True)

                elif method.__class__.__name__ == 'MABAC':
                    pref = method(matrix, weights_copy, types)
                    rank = rank_preferences(pref, reverse = True)

                elif method.__class__.__name__ == 'MULTIMOORA':
                    rank = method(matrix, weights_copy, types)

                elif method.__class__.__name__ == 'WASPAS':
                    pref = method(matrix, weights_copy, types)
                    rank = rank_preferences(pref, reverse = True)
                
                # Assign calculated ranking to column with value of j criterion weight change in `data_sens` dataframe
                data_sens['{:.0f}'.format(dir * change_val * 100) + '%'] = rank

        # Drop column with `-0%` name for visualization result
        if '-0%' in list(data_sens.columns):
            data_sens = data_sens.drop(['-0%'], axis = 1)
        # Set index with alternatives symbols in dataframe with sensitivity analysis results (rankings)
        data_sens = data_sens.set_index('Ai')
        return data_sens