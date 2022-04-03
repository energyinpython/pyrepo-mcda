import numpy as np
import pandas as pd

from .additions import rank_preferences


class Sensitivity_analysis_weights_values():
    def __init__(self):
        """
        Create object of `Sensitivity_analysis_weights_values` method.
        """
        pass


    def __call__(self, matrix, weight_values, types, method, j):
        """
        Method for sensitivity analysis. This method determines rankings of alternatives using chosen
        MCDA method name `mcda_name` for the value of criterion `j` weight set as chosen `weight_value`.

        Parameters
        -----------
            matrix : ndarray
                Decision matrix with performance values of alternatives. This matrix includes
                data on alternatives in rows considering criteria in columns.

            weight_values : ndarray
                Vector with values to be set as the weight of chosen criterion in the sensitivity analysis procedure in range from 0 to 1.

            types : ndarray
                Vector with criteria types. Types must be equal to 1 for profit criteria and -1
                for cost criteria.

            method : class
                Initialized object of class of chosen MCDA method

            j : int
                Index of the column in decision matrix `matrix` that indicates for which criterion
                the weight is set with chosen value.

        Returns
        --------
            data_sens : DataFrame
                dataframe with rankings calculated for subsequent changes of criterion `j` weight.
                Particular rankings for different weight values of criterion `j` are included in
                subsequent columns of the dataframe.

        Examples
        ----------
        >>> sensitivity_analysis = Sensitivity_analysis_weights_values()
        >>> df_sens = sensitivity_analysis(matrix, weight_values, types, method, j)
        """
        list_alt_names = [r'$A_{' + str(i) + '}$' for i in range(1, matrix.shape[0] + 1)]
        return Sensitivity_analysis_weights_values._sensitivity_analysis_weights_values(self, matrix, weight_values, types, method, list_alt_names, j)


    def _change_weights(self, matrix, weight_value, j):
        """
        Method for criteria weights changing in sensitivity analysis procedure.

        Parameters
        -----------
            matrix : ndarray
                Decision matrix with performance values of alternatives. This matrix includes
                data on alternatives in rows considering criteria in columns

            weight_value : float
                Value in range from 0 to 1 to be set as the weight of chosen criterion with index `j`.

            j : int
                Index of the column in decision matrix `matrix` indicating for which criterion
                the weight value is changed.

        Returns
        --------
            new_weights : ndarray
                Vector with criteria weights after changing their values for sensitivity analysis
        """
        
        new_weights = np.ones(matrix.shape[1]) * ((1 - weight_value) / (matrix.shape[1] - 1))
        new_weights[j] = weight_value
        return new_weights


    @staticmethod
    def _sensitivity_analysis_weights_values(self, matrix, weight_values, types, method, list_alt_names, j):
        # Create a dataframe for sensitivity analysis results (rankings)
        data_sens = pd.DataFrame()
        # Assisgn indexes (alternatives symbols) to dataframe `data_sens`
        data_sens['Ai'] = list_alt_names
        
        # Iterate by values of weight change in vector `weight_values`
        for val in weight_values:
            # Change weights using method named `_change_weights` from class `Sensitivity_analysis_weights_values`
            new_weights = self._change_weights(matrix, val, j)

            # Calculate alternatives ranking using selected MCDA method, `matrix`, vector of new weights
            # `new_weights` and criteria types `types`
            if method.__class__.__name__ == 'TOPSIS':
                pref = method(matrix, new_weights, types)
                rank = rank_preferences(pref, reverse = True)

            elif method.__class__.__name__ == 'CODAS':
                pref = method(matrix, new_weights, types)
                rank = rank_preferences(pref, reverse = True)

            elif method.__class__.__name__ == 'VIKOR':
                pref = method(matrix, new_weights, types)
                rank = rank_preferences(pref, reverse = False)

            elif method.__class__.__name__ == 'SPOTIS':
                bounds_min = np.amin(matrix, axis = 0)
                bounds_max = np.amax(matrix, axis = 0)
                bounds = np.vstack((bounds_min, bounds_max))
                pref = method(matrix, new_weights, types, bounds)
                rank = rank_preferences(pref, reverse = False)

            elif method.__class__.__name__ == 'EDAS':
                pref = method(matrix, new_weights, types)
                rank = rank_preferences(pref, reverse = True)

            elif method.__class__.__name__ == 'MABAC':
                pref = method(matrix, new_weights, types)
                rank = rank_preferences(pref, reverse = True)

            elif method.__class__.__name__ == 'MULTIMOORA':
                rank = method(matrix, new_weights, types)

            elif method.__class__.__name__ == 'WASPAS':
                pref = method(matrix, new_weights, types)
                rank = rank_preferences(pref, reverse = True)
            
            # Assign calculated ranking to column with value of j criterion weight change in `data_sens` dataframe
            data_sens['{:.2f}'.format(val)] = rank

        # Set index with alternatives symbols in dataframe with sensitivity analysis results (rankings)
        data_sens = data_sens.set_index('Ai')
        return data_sens