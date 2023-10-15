import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy

from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda.mcda_methods import VMCM, TOPSIS, EDAS, VIKOR, PROMETHEE_II, PROSA_C, COCOSO
from pyrepo_mcda import normalizations as norms
from pyrepo_mcda import correlations as corrs


# heat maps with correlations
def draw_heatmap(df_new_heatmap, title):
    """
    Visualization method to display heatmap with correlations of compared rankings generated using different methods
    
    Parameters
    ----------
        data : DataFrame
            DataFrame with correlation values between compared rankings
        
    Examples
    ---------
    >>> draw_heatmap(df_new_heatmap)
    """
    
    plt.figure(figsize = (11, 5))
    sns.set(font_scale = 1.2)
    heatmap = sns.heatmap(df_new_heatmap, annot=True, fmt=".4f", cmap="YlGnBu",
                          linewidth=0.5, linecolor='w')
    plt.yticks(va="center")
    plt.yticks(va="center")
    plt.xlabel('MCDA methods')
    plt.ylabel('MCDA methods')
    plt.title('Correlation coefficient: ' + title)
    plt.tight_layout()
    plt.savefig('./results_update2/heatmap.pdf')
    plt.savefig('./results_update2/heatmap.eps')
    plt.show()


# Create dictionary class
class Create_dictionary(dict):
  
    # __init__ function
    def __init__(self):
        self = dict()
          
    # Function to add key:value
    def add(self, key, value):
        self[key] = value



def main():
    # Load data including decision matrix and criteria types
    data = pd.read_csv('dataset_waste.csv', index_col = 'Alternative')
    df_data = data.iloc[:len(data) - 1, :]
    types = data.iloc[len(data) - 1, :].to_numpy()

    matrix = df_data.to_numpy()

    # Initialize the VMCM method object
    vmcm = VMCM()

    # Print the criteria to be eliminated
    vmcm._elimination(matrix)

    # Determine criteria weights
    weights = vmcm._weighting(matrix)

    df_vmcm = pd.DataFrame(index=df_data.index)

    # Determine pattern and anti-pattern
    pattern, antipattern = vmcm._pattern_determination(matrix, types)

    # Calculate value of the synthetic measure for each object
    pref = vmcm(matrix, weights, types, pattern, antipattern)

    # Classify evaluated objects according to synthetic measure values
    classes = vmcm._classification(pref)

    # Rank evaluated objects according to synthetic measure values
    rank = rank_preferences(pref, reverse = True)

    df_vmcm['Synthetic measure'] = pref
    df_vmcm['Class'] = classes
    df_vmcm['Rank'] = rank

    df_vmcm.to_csv('./results_update2/vmcm_results.csv')

    # COMAPRATIVE ANALYSIS
    df_comparative = pd.DataFrame(index=df_data.index)
    df_comparative['VMCM'] = rank

    topsis = TOPSIS(normalization_method=norms.minmax_normalization)
    pref = topsis(matrix, weights, types)
    rank = rank_preferences(pref, reverse = True)
    df_comparative['TOPSIS'] = rank

    vikor = VIKOR()
    pref = vikor(matrix, weights, types)
    rank = rank_preferences(pref, reverse = False)
    df_comparative['VIKOR'] = rank

    edas = EDAS()
    pref = edas(matrix, weights, types)
    rank = rank_preferences(pref, reverse = True)
    df_comparative['EDAS'] = rank

    promethee_II = PROMETHEE_II()
    preference_functions = [promethee_II._linear_function for pf in range(len(weights))]
    pref = promethee_II(matrix, weights, types, preference_functions)
    rank = rank_preferences(pref, reverse=True)
    df_comparative['PROMETHEE II'] = rank

    prosa_c = PROSA_C()
    pref = prosa_c(matrix, weights, types, preference_functions)
    rank = rank_preferences(pref, reverse=True)
    df_comparative['PROSA C'] = rank

    cocoso = COCOSO(normalization_method=norms.minmax_normalization)
    pref = cocoso(matrix, weights, types)
    rank = rank_preferences(pref, reverse=True)
    df_comparative['COCOSO'] = rank

    df_comparative.to_csv('./results_update2/df_comparative.csv')


    # Rankings correlations
    results = copy.deepcopy(df_comparative)
    method_types = list(results.columns)
    dict_new_heatmap_rw = Create_dictionary()

    for el in method_types:
        dict_new_heatmap_rw.add(el, [])

    # heatmaps for correlations coefficients
    for i, j in [(i, j) for i in method_types[::-1] for j in method_types]:
        dict_new_heatmap_rw[j].append(corrs.weighted_spearman(results[i], results[j]))

    df_new_heatmap_rw = pd.DataFrame(dict_new_heatmap_rw, index = method_types[::-1])
    df_new_heatmap_rw.columns = method_types

    # correlation matrix with rw coefficient
    draw_heatmap(df_new_heatmap_rw, r'$r_w$')


if __name__ == '__main__':
    main()