import copy
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from pyrepo_mcda.mcda_methods import VIKOR
from pyrepo_mcda.mcda_methods import VIKOR_SMAA
from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda import correlations as corrs
from pyrepo_mcda import normalizations as norms
from pyrepo_mcda import weighting_methods as mcda_weights


# Functions for visualizations

def plot_barplot(df_plot, x_name, y_name, title):
    """
    Display stacked column chart of weights for criteria for `x_name == Weighting methods`
    and column chart of ranks for alternatives `x_name == Alternatives`
    Parameters
    ----------
        df_plot : dataframe
            dataframe with criteria weights calculated different weighting methods
            or with alternaives rankings for different weighting methods
        x_name : str
            name of x axis, Alternatives or Weighting methods
        y_name : str
            name of y axis, Ranks or Weight values
        title : str
            name of chart title, Weighting methods or Criteria
    Examples
    ----------
    >>> plot_barplot(df_plot, x_name, y_name, title)
    """
    
    list_rank = np.arange(1, len(df_plot) + 1, 1)
    stacked = True
    width = 0.5
    if x_name == 'Alternatives':
        stacked = False
        width = 0.8
    elif x_name == 'Alternative':
        pass
    else:
        df_plot = df_plot.T
    ax = df_plot.plot(kind='bar', width = width, stacked=stacked, edgecolor = 'black', figsize = (9,4))
    ax.set_xlabel(x_name, fontsize = 12)
    ax.set_ylabel(y_name, fontsize = 12)

    if x_name == 'Alternatives':
        ax.set_yticks(list_rank)

    ax.set_xticklabels(df_plot.index, rotation = 'horizontal')
    ax.tick_params(axis = 'both', labelsize = 12)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=4, mode="expand", borderaxespad=0., edgecolor = 'black', title = title, fontsize = 11)

    ax.grid(True, linestyle = '--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()


def draw_heatmap(data, title):
    """
    Display heatmap with correlations of compared rankings generated using different methods
    Parameters
    ----------
        data : dataframe
            dataframe with correlation values between compared rankings
        title : str
            title of chart containing name of used correlation coefficient
    Examples
    ----------
    >>> draw_heatmap(data, title)
    """

    plt.figure(figsize = (6, 4))
    sns.set(font_scale=1.0)
    heatmap = sns.heatmap(data, annot=True, fmt=".2f", cmap="RdYlBu",
                          linewidth=0.5, linecolor='w')
    plt.yticks(va="center")
    plt.xlabel('Weighting methods')
    plt.title('Correlation coefficient: ' + title)
    plt.tight_layout()
    plt.show()

def draw_heatmap_smaa(data, title):
    """
    Display heatmap with correlations of compared rankings generated using different methods
    Parameters
    ----------
        data : dataframe
            dataframe with correlation values between compared rankings
        title : str
            title of chart containing name of used correlation coefficient
    Examples
    ----------
    >>> draw_heatmap(data, title)
    """

    sns.set(font_scale=1.0)
    heatmap = sns.heatmap(data, annot=True, fmt=".2f", cmap="RdYlBu_r",
                        linewidth=0.05, linecolor='w')
    plt.yticks(rotation=0)
    plt.ylabel('Alternatives')
    plt.tick_params(labelbottom=False,labeltop=True)
    
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_boxplot(data):
    """
    Display boxplot showing distribution of criteria weights determined with different methods.
    Parameters
    ----------
        data : dataframe
            dataframe with correlation values between compared rankings
    Examples
    ---------
    >>> plot_boxplot(data)
    """
    
    df_melted = pd.melt(data)
    plt.figure(figsize = (7, 4))
    ax = sns.boxplot(x = 'variable', y = 'value', data = df_melted, width = 0.6)
    ax.grid(True, linestyle = '--')
    ax.set_axisbelow(True)
    ax.set_xlabel('Criterion', fontsize = 12)
    ax.set_ylabel('Different weights distribution', fontsize = 12)
    plt.tight_layout()
    plt.show()


# Create dictionary class
class Create_dictionary(dict):
  
    # __init__ function
    def __init__(self):
        self = dict()
          
    # Function to add key:value
    def add(self, key, value):
        self[key] = value



# main
def main():

    matrix = np.array([[155.3, 74.0, 340, 673, 456.0, 111, 115, 106, 244, 39.8, 65440],
        [162.2, 79.5, 247, 639, 283.0, 113, 118, 107, 263, 38.8, 60440],
        [112.5, 68.0, 198, 430, 266.0, 98, 105, 91, 230, 38.1, 56575],
        [90.1, 66.0, 150, 360, 201.2, 120, 131, 109, 259, 34.8, 32495],
        [99.4, 77.0, 150, 310, 201.2, 97, 102, 90, 260, 36.4, 45635],
        [89.5, 40.0, 110, 320, 147.5, 111, 123, 99, 226, 34.8, 28425],
        [124.3, 95.0, 125, 247, 187.7, 78, 78, 77, 222, 40.0, 84595],
        [155.3, 79.2, 160, 300, 214.6, 79, 79, 80, 227, 38.4, 105150],
        [162.2, 100.0, 205, 420, 502.9, 120, 124, 115, 402, 40.3, 96440],
        [96.3, 39.2, 100, 395, 134.1, 120, 132, 108, 258, 34.8, 35245],
        [162.2, 100.0, 205, 420, 502.9, 98, 103, 93, 371, 40.8, 127940],
        [102.5, 38.3, 101, 295, 136.1, 133, 145, 121, 170, 34.8, 34250]])

    types = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1])

    

    # matrix = np.array([[80, 16, 2, 5],
    # [110, 32, 2, 9],
    # [130, 64, 4, 9],
    # [185, 64, 4, 1],
    # [135, 64, 3, 4],
    # [140, 32, 3, 5],
    # [185, 64, 6, 7],
    # [110, 16, 3, 3],
    # [120, 16, 4, 3],
    # [340, 128, 6, 5]])

    # types = np.array([-1, 1, 1, 1])

    # Symbols for alternatives Ai
    list_alt_names = [r'$A_{' + str(i) + '}$' for i in range(1, matrix.shape[0] + 1)]
    # Symbols for columns Cj
    cols = [r'$C_{' + str(j) + '}$' for j in range(1, matrix.shape[1] + 1)]

    
    # Part 1 - study with single weighting method
    
    # Determine criteria weights with chosen weighting method
    weights = mcda_weights.entropy_weighting(matrix)

    # Create the VIKOR method object
    # vikor = VIKOR(normalization_method=norms.minmax_normalization)
    vikor = VIKOR(normalization_method=norms.minmax_normalization)
    
    # Calculate alternatives preference function values with VIKOR method
    pref = vikor(matrix, weights, types)

    # when there is only one (single) preference vector
    rank = rank_preferences(pref, reverse = False)

    print(rank)

    
    # Part 2 - study with several weighting methods
    # Create a list with weighting methods that you want to explore
    weighting_methods_set = [
        mcda_weights.entropy_weighting,
        # mcda_weights.std_weighting,
        mcda_weights.critic_weighting,
        mcda_weights.gini_weighting,
        mcda_weights.merec_weighting,
        mcda_weights.stat_var_weighting,
        # mcda_weights.cilos_weighting,
        mcda_weights.idocriw_weighting,
        mcda_weights.angle_weighting,
        mcda_weights.coeff_var_weighting
    ]
    

    #df_weights = pd.DataFrame(weights.reshape(1, -1), index = ['Weights'], columns = cols)
    # Create dataframes for weights, preference function values and rankings determined using different weighting methods
    df_weights = pd.DataFrame(index = cols)
    df_preferences = pd.DataFrame(index = list_alt_names)
    df_rankings = pd.DataFrame(index = list_alt_names)

    # Create the VIKOR method object
    vikor = VIKOR(normalization_method=norms.minmax_normalization)

    for weight_type in weighting_methods_set:
        
        if weight_type.__name__ in ["cilos_weighting", "idocriw_weighting", "angle_weighting", "merec_weighting"]:
            weights = weight_type(matrix, types)
        else:
            weights = weight_type(matrix)

        df_weights[weight_type.__name__[:-10].upper().replace('_', ' ')] = weights
        pref = vikor(matrix, weights, types)

        rank = rank_preferences(pref, reverse = False)

        df_preferences[weight_type.__name__[:-10].upper().replace('_', ' ')] = pref
        df_rankings[weight_type.__name__[:-10].upper().replace('_', ' ')] = rank
        

    
    # plot criteria weights distribution using box chart
    plot_boxplot(df_weights.T)

    # plot stacked column chart of criteria weights
    plot_barplot(df_weights, 'Weighting methods', 'Weight value', 'Criteria')

    # plot column chart of alternatives rankings
    plot_barplot(df_rankings, 'Alternatives', 'Rank', 'Weighting methods')

    # Plot heatmaps of rankings correlation coefficient
    # Create dataframe with rankings correlation values
    results = copy.deepcopy(df_rankings)
    method_types = list(results.columns)
    dict_new_heatmap_rw = Create_dictionary()

    for el in method_types:
        dict_new_heatmap_rw.add(el, [])

    for i, j in [(i, j) for i in method_types[::-1] for j in method_types]:
        dict_new_heatmap_rw[j].append(corrs.weighted_spearman(results[i], results[j]))
            
    df_new_heatmap_rw = pd.DataFrame(dict_new_heatmap_rw, index = method_types[::-1])
    df_new_heatmap_rw.columns = method_types

    # Plot heatmap with rankings correlation
    draw_heatmap(df_new_heatmap_rw, r'$r_w$')
    

    
    # SMAA method
    cols_ai = [str(el) for el in range(1, matrix.shape[0] + 1)]

    # criteria number
    n = matrix.shape[1]
    # SMAA iterations number (number of weight vectors for SMAA)
    iterations = 10000

    start = time.time()
    # create the VIKOR_SMAA object
    vikor_smaa = VIKOR_SMAA()
    # generate matrix with weight vectors for SMAA
    weight_vectors = vikor_smaa._generate_weights(n, iterations)

    # run the VIKOR_SMAA method
    rank_acceptability_index, central_weight_vector, rank_scores = vikor_smaa(matrix, weight_vectors, types)

    end = time.time() - start
    print('Run time: ', end)
    
    acc_in_df = pd.DataFrame(rank_acceptability_index, index = list_alt_names, columns = cols_ai)
    

    matplotlib.rcdefaults()
    plot_barplot(acc_in_df, 'Alternative', 'Rank acceptability index', 'Rank')

    draw_heatmap_smaa(acc_in_df, 'Rank acceptability indexes')

    central_weights_df = pd.DataFrame(central_weight_vector, index = list_alt_names, columns = cols)
    

    rank_scores_df = pd.DataFrame(rank_scores, index = list_alt_names, columns = ['Rank'])
    
    


if __name__ == '__main__':
    main()