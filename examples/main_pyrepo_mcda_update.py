import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib

# example 1 involving the decision problem of selecting the best wind farm location based on:
"""
Ziemba, P., Wątróbski, J., Zioło, M., & Karczmarczyk, A. (2017). 
Using the PROSA method in offshore wind farm location problems. Energies, 10(11), 1755.
DOI: https://doi.org/10.3390/en10111755
"""

# example 2 involving the decision problem of selecting the best wind farm location based on:
"""
Kizielewicz, B., Wątróbski, J., & Sałabun, W. (2020). Identification of relevant criteria set 
in the MCDA process—Wind farm location case study. Energies, 13(24), 6548.
"""

from pyrepo_mcda.mcda_methods import CRADIS, AHP, MARCOS, SAW, ARAS, COPRAS, PROMETHEE_II, PROSA_C
from pyrepo_mcda.mcda_methods import TOPSIS, VIKOR, MABAC, EDAS, SPOTIS, WASPAS

from pyrepo_mcda import normalizations as norms
from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda import correlations as corrs
from pyrepo_mcda import weighting_methods as mcda_weights




# bar (column) chart
def plot_barplot(df_plot, legend_title, num):
    """
    Visualization method to display column chart of alternatives rankings obtained with 
    different methods.

    Parameters
    ----------
        df_plot : DataFrame
            DataFrame containing rankings of alternatives obtained with different methods.
            The particular rankings are included in subsequent columns of DataFrame.
        title : str
            Title of the legend (Name of group of explored methods, for example MCDA methods or Distance metrics).
    
    Examples
    ----------
    >>> plot_barplot(df_plot, legend_title='MCDA methods')
    """
    step = 1
    list_rank = np.arange(1, len(df_plot) + 1, step)

    ax = df_plot.plot(kind='bar', width = 0.8, stacked=False, edgecolor = 'black', figsize = (9,4))
    ax.set_xlabel('Alternatives', fontsize = 12)
    ax.set_ylabel('Rank', fontsize = 12)
    ax.set_yticks(list_rank)

    ax.set_xticklabels(df_plot.index, rotation = 'horizontal')
    ax.tick_params(axis = 'both', labelsize = 12)
    y_ticks = ax.yaxis.get_major_ticks()
    ax.set_ylim(0, len(df_plot) + 1)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=4, mode="expand", borderaxespad=0., edgecolor = 'black', title = legend_title, fontsize = 12)

    ax.grid(True, linestyle = ':')
    ax.set_axisbelow(True)
    plt.tight_layout()
    legend_title = legend_title.replace("$", "")
    plt.savefig('./results/' + 'bar_chart_' + legend_title + str(num) + '.eps')
    plt.savefig('./results/' + 'bar_chart_' + legend_title + str(num) + '.png')
    plt.show()


# heat maps with correlations
def draw_heatmap(df_new_heatmap, title, num):
    """
    Visualization method to display heatmap with correlations of compared rankings generated using different methods
    
    Parameters
    ----------
        data : DataFrame
            DataFrame with correlation values between compared rankings
        title : str
            title of chart containing name of used correlation coefficient
    
    Examples
    ---------
    >>> draw_heatmap(df_new_heatmap, title)
    """
    plt.figure(figsize = (8, 5))
    sns.set(font_scale = 1.2)
    heatmap = sns.heatmap(df_new_heatmap, annot=True, fmt=".3f", cmap="RdYlGn",
                          linewidth=0.5, linecolor='w')
    plt.yticks(va="center")
    plt.xlabel('MCDA methods')
    plt.title('Correlation: ' + title)
    plt.tight_layout()
    title = title.replace("$", "")
    plt.savefig('./results/' + 'correlations_' + title + str(num) + '.eps')
    plt.savefig('./results/' + 'correlations_' + title + str(num) + '.png')
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

    # Example 1
    # Decision matrix
    matrix = np.array([[16347, 14219, 8160, 8160],
    [9, 8.5, 9, 8.5],
    [73.8, 55, 64.8, 62.5],
    [36.7, 36, 28.5, 29.5],
    [1.5, 2, 2, 1.5],
    [3730, 3240, 1860, 1860],
    [2, 1, 2, 1],
    [1, 1, 2, 3],
    [38.8, 33.1, 45.8, 27.3],
    [4, 2, 4, 3],
    [1720524, 1496512, 858830, 858830],
    [40012, 34803, 19973, 19973]])

    # Transpose the decision matrix to place alternatives in rows and criteria in columns
    matrix = matrix.T

    # Provide weights
    weights = np.array([20, 5, 5, 1.67, 1.67, 11.67, 11.67, 5, 5, 16.67, 8.33, 8.33])
    # Normalize weight values using sum normalization
    # All weights must sum to 1
    weights = weights / np.sum(weights)

    # provide criteria types. 1 represents profit criteria and -1 denotes cost criteria.
    # Other values are not allowed.
    types = np.array([-1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1])

    # Rows names
    alt_names = [r'$A_{' + str(el) + '}$' for el in range(1, matrix.shape[0] + 1)]
    # Columns names
    col_names = [r'$C_{' + str(el) + '}$' for el in range(1, matrix.shape[1] + 1)]
    # save the DataFrame with dataset to csv file
    df_ex1 = pd.DataFrame(matrix, index = alt_names, columns = col_names)
    df_ex1 = df_ex1.rename_axis('Ai')
    df_ex1.to_csv('./results/dataset1.csv')

    # Create the DataFrame for rankings
    rank_results = pd.DataFrame(index=alt_names)

    
    # PROMETHEE II
    p = np.array([7280, 4, 13.4, 7.4, 3, 1662, 3, 3, 13.8, 3, 766240, 17820])
    promethee_II = PROMETHEE_II()
    preference_functions = [promethee_II._vshape_function for pf in range(len(weights))]

    pref = promethee_II(matrix, weights, types, preference_functions, p = p)
    rank = rank_preferences(pref, reverse=True)
    rank_results['PROMETHEE II'] = rank

    # PROSA-C
    s = np.repeat(0.3, len(weights))
    prosa_c = PROSA_C()
    pref = prosa_c(matrix, weights, types, preference_functions, p = p, s = s)
    rank = rank_preferences(pref, reverse=True)
    rank_results['PROSA C'] = rank

    # ARAS sum norm
    aras = ARAS(normalization_method=norms.linear_normalization)
    pref = aras(matrix, weights, types)
    rank = rank_preferences(pref, reverse=True)
    rank_results['ARAS'] = rank

    # COPRAS sum norm
    copras = COPRAS()
    pref = copras(matrix, weights, types)
    rank = rank_preferences(pref, reverse=True)
    rank_results['COPRAS'] = rank

    # CRADIS linear norm
    cradis = CRADIS()
    pref = cradis(matrix, weights, types)
    rank = rank_preferences(pref, reverse=True)
    rank_results['CRADIS'] = rank

    # MARCOS
    marcos = MARCOS()
    pref = marcos(matrix, weights, types)
    rank = rank_preferences(pref, reverse=True)
    rank_results['MARCOS'] = rank

    # SAW linear norm
    saw = SAW()
    pref = saw(matrix, weights, types)
    rank = rank_preferences(pref, reverse=True)
    rank_results['SAW'] = rank

    # VIKOR
    vikor = VIKOR()
    pref = vikor(matrix, weights, types)
    rank = rank_preferences(pref, reverse=False)
    rank_results['VIKOR'] = rank
    rank_results = rank_results.rename_axis('Ai')
    rank_results.to_csv('./results/results_1.csv')


    print(rank_results)
    plot_barplot(rank_results, legend_title='MCDA methods', num = 1)



    method_types = list(rank_results.columns)
    dict_new_heatmap_rw = Create_dictionary()
    for el in method_types:
        dict_new_heatmap_rw.add(el, [])


    # heatmaps for correlations coefficients
    for i, j in [(i, j) for i in method_types[::-1] for j in method_types]:
        dict_new_heatmap_rw[j].append(corrs.weighted_spearman(rank_results[i], rank_results[j]))

    df_new_heatmap_rw = pd.DataFrame(dict_new_heatmap_rw, index = method_types[::-1])
    df_new_heatmap_rw.columns = method_types

    # correlation matrix with rw coefficient
    draw_heatmap(df_new_heatmap_rw, r'$r_w$', num = 1)

    # Example 2

    matrix = np.array([[106.78, 6.75, 2.00, 220, 6.00, 1, 52.00, 455.50, 8.90, 36.80],
    [86.37, 7.12, 3.00, 400, 10.00, 0, 20.00, 336.50, 7.20, 29.80],
    [104.85, 6.95, 60.00, 220, 7.00, 1, 60.00, 416.00, 8.70, 36.20],
    [46.60, 6.04, 1.00, 220, 3.00, 0, 50.00, 277.00, 3.90, 16.00],
    [69.18, 7.05, 33.16, 220, 8.00, 0, 35.49, 364.79, 5.39, 33.71],
    [66.48, 6.06, 26.32, 220, 6.53, 0, 34.82, 304.02, 4.67, 27.07],
    [74.48, 6.61, 48.25, 400, 4.76, 1, 44.19, 349.45, 4.93, 28.89],
    [73.67, 6.06, 19.54, 400, 3.19, 0, 46.41, 354.65, 8.01, 21.09],
    [100.58, 6.37, 39.27, 220, 8.43, 1, 22.07, 449.42, 7.89, 17.62],
    [94.81, 6.13, 50.58, 220, 4.18, 1, 21.14, 450.88, 5.12, 17.30],
    [48.93, 7.12, 21.48, 220, 5.47, 1, 55.72, 454.71, 8.39, 19.16],
    [74.75, 6.58, 7.08, 400, 9.90, 1, 26.01, 455.17, 4.78, 18.44]])

    types = np.array([1, 1, -1, 1, -1, -1, 1, -1, -1, 1])

    weights = mcda_weights.entropy_weighting(matrix)

    alt_names = [r'$A_{' + str(el) + '}$' for el in range(1, matrix.shape[0] + 1)]
    col_names = [r'$C_{' + str(el) + '}$' for el in range(1, matrix.shape[1] + 1)]
    df_ex2 = pd.DataFrame(matrix, index = alt_names, columns = col_names)
    df_ex2 = df_ex2.rename_axis('Ai')
    df_ex2.to_csv('./results/dataset2.csv')

    rank_results = pd.DataFrame(index=alt_names)

    promethee_II = PROMETHEE_II()
    preference_functions = [prosa_c._vshape_function for pf in range(len(weights))]

    pref = promethee_II(matrix, weights, types, preference_functions)
    rank = rank_preferences(pref, reverse=True)
    rank_results['PROMETHEE II'] = rank

    prosa_c = PROSA_C()

    u = np.sqrt(np.sum(np.square(np.mean(matrix, axis = 0) - matrix), axis = 0) / matrix.shape[0])
    p = 2 * u
    q = 0.5 * u
    s = np.repeat(0.3, len(weights))

    pref = prosa_c(matrix, weights, types, preference_functions, p = p, q = q, s = s)
    rank = rank_preferences(pref, reverse=True)
    rank_results['PROSA C'] = rank

    topsis = TOPSIS()
    pref = topsis(matrix, weights, types)
    rank = rank_preferences(pref, reverse=True)
    rank_results['TOPSIS'] = rank

    vikor = VIKOR()
    pref = vikor(matrix, weights, types)
    rank = rank_preferences(pref, reverse=False)
    rank_results['VIKOR'] = rank

    # SPOTIS preferences must be sorted in ascending order
    bounds_min = np.amin(matrix, axis = 0)
    bounds_max = np.amax(matrix, axis = 0)
    bounds = np.vstack((bounds_min, bounds_max))
    spotis = SPOTIS()
    pref = spotis(matrix, weights, types, bounds)
    rank = rank_preferences(pref, reverse = False)
    rank_results['SPOTIS'] = rank

    # EDAS preferences must be sorted in descending order
    edas = EDAS()
    pref = edas(matrix, weights, types)
    rank = rank_preferences(pref, reverse = True)
    rank_results['EDAS'] = rank

    # MABAC preferences must be sorted in descending order
    mabac = MABAC(normalization_method = norms.minmax_normalization)
    pref = mabac(matrix, weights, types)
    rank = rank_preferences(pref, reverse = True)
    rank_results['MABAC'] = rank

    # CRADIS linear norm
    cradis = CRADIS(normalization_method=norms.minmax_normalization)
    pref = cradis(matrix, weights, types)
    rank = rank_preferences(pref, reverse=True)
    rank_results['CRADIS'] = rank
    rank_results = rank_results.rename_axis('Ai')
    rank_results.to_csv('./results/results_2.csv')

    print(rank_results)
    matplotlib.rc_file_defaults()
    plot_barplot(rank_results, legend_title='MCDA methods', num = 2)


    method_types = list(rank_results.columns)
    dict_new_heatmap_rw = Create_dictionary()
    for el in method_types:
        dict_new_heatmap_rw.add(el, [])


    # heatmaps for correlations coefficients
    for i, j in [(i, j) for i in method_types[::-1] for j in method_types]:
        dict_new_heatmap_rw[j].append(corrs.weighted_spearman(rank_results[i], rank_results[j]))

    df_new_heatmap_rw = pd.DataFrame(dict_new_heatmap_rw, index = method_types[::-1])
    df_new_heatmap_rw.columns = method_types

    # correlation matrix with rw coefficient
    draw_heatmap(df_new_heatmap_rw, r'$r_w$', num = 2)




if __name__ == '__main__':
    main()


