import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy

from pyrepo_mcda.mcda_methods import TOPSIS, VIKOR, CODAS, SPOTIS, PROMETHEE_II, COPRAS
from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda import weighting_methods as mcda_weights
from pyrepo_mcda import correlations as corrs
from pyrepo_mcda import normalizations as norms
from pyrepo_mcda import distance_metrics as dists

# Create dictionary class to create a data frame with correlation values
class Create_dictionary(dict):
  
    # __init__ function
    def __init__(self):
        self = dict()
          
    # Function to add key:value
    def add(self, key, value):
        self[key] = value

# bar (column) chart
def plot_barplot(df_plot, stacked = False):
    """
    Visualization method to display column chart of alternatives rankings obtained with 
    different methods.

    Parameters
    ----------
        df_plot : DataFrame
            DataFrame containing rankings of alternatives obtained with different methods.
            The particular rankings are included in subsequent columns of DataFrame.

        stacked : Boolean
            Variable denoting if the chart is to be stacked or not.

    Examples
    ----------
    >>> plot_barplot(df_plot)
    """

    ax = df_plot.plot(kind='bar', width = 0.8, stacked=stacked, edgecolor = 'black', figsize = (9,4))
    if stacked == False:
        step = 1
        list_rank = np.arange(1, len(df_plot) + 1, step)
        ax.set_yticks(list_rank)
        ax.set_ylim(0, len(df_plot) + 1)

    ax.set_xticklabels(df_plot.index, rotation = 'horizontal')
    ax.tick_params(axis = 'both', labelsize = 12)
    y_ticks = ax.yaxis.get_major_ticks()
    
    ncol = df_plot.shape[1]
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol = ncol, mode="expand", borderaxespad=0., edgecolor = 'black', fontsize = 12)

    ax.grid(True)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()


# plot line chart
def plot_lineplot(df_plot):
    """
    Visualization method to display line chart of alternatives rankings.

    Parameters
    ----------
        df_plot : DataFrame
            DataFrame containing rankings of alternatives obtained with different weight of
            selected criterion. The particular rankings are contained in subsequent columns of 
            DataFrame.
        
    Examples
    ----------
    >>> plot_lineplot(df_plot)
    """

    plt.figure(figsize = (9, 4))
    for j in range(df_plot.shape[0]):
        
        plt.plot(df_plot.iloc[j, :], '-o', linewidth = 2)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        plt.annotate(df_plot.index[j], (x_max, df_plot.iloc[j, -1]),
                        fontsize = 12, style='italic',
                        horizontalalignment='left')

    plt.xlabel("Method", fontsize = 12)
    plt.ylabel("Rank", fontsize = 12)
    plt.yticks(np.arange(0, df_plot.shape[0]) + 1, fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# plot radar chart
def plot_radar(data):
    """
    Visualization method to display rankings of alternatives obtained with different methods
    on the radar chart.

    Parameters
    -----------
        data : DataFrame
            DataFrame containing containing rankings of alternatives obtained with different 
            methods. The particular rankings are contained in subsequent columns of DataFrame.
        
    Examples
    ----------
    >>> plot_radar(data)
    """

    fig=plt.figure()
    ax = fig.add_subplot(111, polar = True)

    for col in list(data.columns):
        labels=np.array(list(data.index))
        stats = data.loc[labels, col].values

        angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        # close the plot
        stats=np.concatenate((stats,[stats[0]]))
        angles=np.concatenate((angles,[angles[0]]))
    
        lista = list(data.index)
        lista.append(data.index[0])
        labels=np.array(lista)

        ax.plot(angles, stats, '-o', linewidth=2)
    
    ax.set_thetagrids(angles * 180/np.pi, labels)
    ax.set_rgrids(np.arange(1, data.shape[0] + 1, 1))
    ax.grid(True)
    ax.set_axisbelow(True)
    # plt.legend(data.columns, bbox_to_anchor=(1.0, 0.95, 0.4, 0.2), loc='upper left')
    if data.shape[1] % 2 == 0:
        ncol = data.shape[1] // 2
    else:
        ncol = data.shape[1] // 2 + 1
    plt.legend(data.columns, bbox_to_anchor=(-0.1, 1.1, 1.2, .102), loc='lower left',
    ncol = ncol, mode="expand", borderaxespad=0., edgecolor = 'black', fontsize = 12)
    plt.tight_layout()
    plt.show()


# plot scatter chart
def plot_scatter(data, model_compare):
    """
    Display scatter plot comparing real and predicted ranking.

    Parameters
    -----------
        data: dataframe
        model_compare : list[list] with two columns including compared rankings

    Examples
    ----------
    >>> plot_scatter(data)
    """
    
    list_rank = np.arange(1, len(data) + 1, 1)
    list_alt_names = data.index
    for it, el in enumerate(model_compare):
        
        xx = [min(data[el[0]]), max(data[el[0]])]
        yy = [min(data[el[1]]), max(data[el[1]])]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(xx, yy, linestyle = '--', zorder = 1)

        ax.scatter(data[el[0]], data[el[1]], marker = 'o', color = 'royalblue', zorder = 2)
        for i, txt in enumerate(list_alt_names):
            ax.annotate(txt, (data[el[0]][i], data[el[1]][i]), fontsize = 14, style='italic',
                         verticalalignment='bottom', horizontalalignment='right')

        ax.set_xlabel(el[0], fontsize = 16)
        ax.set_ylabel(el[1], fontsize = 16)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_xticks(list_rank)
        ax.set_yticks(list_rank)

        x_ticks = ax.xaxis.get_major_ticks()
        y_ticks = ax.yaxis.get_major_ticks()

        ax.set_xlim(0.5, len(data) + 0.5)
        ax.set_ylim(0.5, len(data) + 0.5)

        ax.grid(True, linestyle = '--')
        ax.set_axisbelow(True)
    
        plt.tight_layout()
        plt.show()


# boxplot chart for visualization of distribution
def plot_boxplot(data):
    """
    Display boxplot showing distribution of preference values determined with different methods.
    Parameters
    ----------
        data : dataframe
            dataframe with correlation values between compared rankings
        
    Examples
    ----------
    >>> plot_boxplot(data, title)
    """
    
    df_melted = pd.melt(data)
    plt.figure(figsize = (7, 4))
    ax = sns.boxplot(x = 'variable', y = 'value', data = df_melted, width = 0.4)
    ax.set_xlabel('', fontsize = 12)
    ax.set_ylabel('', fontsize = 12)
    ax.grid(True, linestyle = '--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()


# heat maps with correlations
def draw_heatmap(df_new_heatmap):
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
    plt.figure(figsize = (8, 5))
    sns.set(font_scale = 1.2)
    heatmap = sns.heatmap(df_new_heatmap, annot=True, fmt=".2f", cmap="PuBu",
                          linewidth=0.5, linecolor='w')
    plt.yticks(va="center")
    plt.tight_layout()
    plt.show()

def main():

    # load decision matrix from csv file

    data = pd.read_csv('dataset_cars.csv', index_col = 'Ai')

    df_data = data.iloc[:len(data) - 1, :]
    types = data.iloc[len(data) - 1, :].to_numpy()
    matrix = df_data.to_numpy()
    weights = mcda_weights.entropy_weighting(matrix)

    list_alt_names = [r'$A_{' + str(i) + '}$' for i in range(1, df_data.shape[0] + 1)]
    cols = [r'$C_{' + str(j) + '}$' for j in range(1, data.shape[1] + 1)]

    # Calculate rankings of different MCDA methods
    rank_results = pd.DataFrame(index=list_alt_names)

    # TOPSIS
    topsis = TOPSIS(normalization_method = norms.minmax_normalization, distance_metric = dists.euclidean)
    pref = topsis(matrix, weights, types)
    rank = rank_preferences(pref, reverse = True)
    rank_results['TOPSIS'] = rank

    # VIKOR
    vikor = VIKOR(normalization_method = norms.minmax_normalization)
    pref = vikor(matrix, weights, types)
    rank = rank_preferences(pref, reverse = False)
    rank_results['VIKOR'] = rank

    # CODAS
    codas = CODAS(normalization_method = norms.linear_normalization, distance_metric = dists.euclidean, tau = 0.02)
    pref = codas(matrix, weights, types)
    rank = rank_preferences(pref, reverse = True)
    rank_results['CODAS'] = rank

    # SPOTIS
    bounds_min = np.amin(matrix, axis = 0)
    bounds_max = np.amax(matrix, axis = 0)
    bounds = np.vstack((bounds_min, bounds_max))
    spotis = SPOTIS()
    pref = spotis(matrix, weights, types, bounds)
    rank = rank_preferences(pref, reverse = False)
    rank_results['SPOTIS'] = rank

    # PROMETHEE II
    promethee_II = PROMETHEE_II()
    pref = promethee_II(matrix, weights, types)
    rank = rank_preferences(pref, reverse=True)
    rank_results['PROMETHEE II'] = rank

    # COPRAS
    copras = COPRAS(normalization_method = norms.sum_normalization)
    pref = copras(matrix, weights, types)
    rank = rank_preferences(pref, reverse=True)
    rank_results['COPRAS'] = rank

    # bar chart
    plot_barplot(rank_results)

    # line chart
    plot_lineplot(rank_results)

    # radar chart
    plot_radar(rank_results)

    # plot scatter chart
    model_compare = []
    names = list(rank_results.columns)
    model_compare = [[names[0], names[1]]]
    plot_scatter(rank_results, model_compare)

    # Criteria weights distribution
    # Create a list with weighting methods that you want to explore
    weighting_methods_set = [
        mcda_weights.equal_weighting,
        mcda_weights.entropy_weighting,
        mcda_weights.critic_weighting,
        mcda_weights.gini_weighting,
        mcda_weights.merec_weighting,
        mcda_weights.stat_var_weighting,
        mcda_weights.idocriw_weighting,
        mcda_weights.angle_weighting,
        mcda_weights.coeff_var_weighting
    ]

    df_weights = pd.DataFrame(index = cols)
    df_preferences = pd.DataFrame(index = list_alt_names)
    df_rankings = pd.DataFrame(index = list_alt_names)

    # Create dataframes for weights, preference function values and rankings determined using different weighting methods
    df_weights = pd.DataFrame(index = cols)
    df_preferences = pd.DataFrame(index = list_alt_names)
    df_rankings = pd.DataFrame(index = list_alt_names)

    # Create the VIKOR method object
    vikor = VIKOR()
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

    # distribution of criteria weights
    plot_boxplot(df_weights.T)
    # bar chart of criteria weights values for different weighting methods
    plot_barplot(df_weights.T, stacked = True)

    # Rankings correlations
    results = copy.deepcopy(rank_results)
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
    draw_heatmap(df_new_heatmap_rw)


if __name__ == '__main__':
    main()