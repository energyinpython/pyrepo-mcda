import numpy as np
import pandas as pd
import copy
import matplotlib
from tabulate import tabulate
from visualizations import *

from pyrepo_mcda.mcda_methods import CODAS, TOPSIS, WASPAS, VIKOR, SPOTIS, EDAS, MABAC, MULTIMOORA

from pyrepo_mcda import distance_metrics as dists
from pyrepo_mcda import correlations as corrs
from pyrepo_mcda import normalizations as norms
from pyrepo_mcda import weighting_methods as mcda_weights
from pyrepo_mcda import compromise_rankings as compromises
from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda.sensitivity_analysis import Sensitivity_analysis_weights


# Create dictionary class
class Create_dictionary(dict):
  
    # __init__ function
    def __init__(self):
        self = dict()
          
    # Function to add key:value
    def add(self, key, value):
        self[key] = value


def main():
    # load name of file with input data
    data = pd.read_csv('data.csv', index_col = 'Ai')

    df_data = data.iloc[:len(data) - 2, :]
    weights = data.iloc[len(data) - 2, :].to_numpy()
    types = data.iloc[len(data) - 1, :].to_numpy()

    print('dataset:')
    header = [df_data.index.name]
    header = header + list(df_data.columns)
    print(tabulate(df_data, headers = header, tablefmt='github'))

    list_alt_names = [r'$A_{' + str(i) + '}$' for i in range(1, df_data.shape[0] + 1)]

    matrix = df_data.to_numpy()

    # Exploration of results for different chosen distance metrics
    distance_metrics = [
        dists.euclidean,
        dists.manhattan,
        dists.hausdorff,
        dists.chebyshev,
        dists.bray_curtis,
        dists.canberra,
        dists.lorentzian,
        dists.jaccard,
        dists.std_euclidean
    ]

    # Create dataframes for preference function values and rankings determined using distance metrics
    df_preferences = pd.DataFrame(index = list_alt_names)
    df_rankings = pd.DataFrame(index = list_alt_names)

    for distance_metric in distance_metrics:
        # Create the TOPSIS method object
        topsis = TOPSIS(normalization_method = norms.minmax_normalization, distance_metric = distance_metric)
        pref = topsis(matrix, weights, types)
        rank = rank_preferences(pref, reverse = True)
        df_preferences[distance_metric.__name__.capitalize().replace('_', ' ')] = pref
        df_rankings[distance_metric.__name__.capitalize().replace('_', ' ')] = rank
        

    print(df_rankings)
    # plot bar chart of alternatives rankings
    plot_barplot(df_rankings, 'Distance metrics')

    # plot box chart of alternatives preference values
    plot_boxplot(df_preferences.T)

    rank_results = pd.DataFrame()
    rank_results['Ai'] = list(list_alt_names)

    # TOPSIS
    # TOPSIS preferences must be sorted in descending order
    topsis = TOPSIS(normalization_method = norms.minmax_normalization, distance_metric = dists.euclidean)
    pref = topsis(matrix, weights, types)
    rank = rank_preferences(pref, reverse = True)
    rank_results['TOPSIS'] = rank


    # CODAS
    # CODAS preferences must be sorted in descending order
    codas = CODAS(normalization_method = norms.linear_normalization, distance_metric = dists.euclidean, tau = 0.02)
    pref = codas(matrix, weights, types)
    rank = rank_preferences(pref, reverse = True)
    rank_results['CODAS'] = rank


    # VIKOR
    # VIKOR preferences must be sorted in ascending order
    vikor = VIKOR(normalization_method = norms.minmax_normalization)
    pref = vikor(matrix, weights, types)
    rank = rank_preferences(pref, reverse = False)
    rank_results['VIKOR'] = rank


    # SPOTIS
    # SPOTIS preferences must be sorted in ascending order
    bounds_min = np.amin(matrix, axis = 0)
    bounds_max = np.amax(matrix, axis = 0)
    bounds = np.vstack((bounds_min, bounds_max))
    spotis = SPOTIS()
    pref = spotis(matrix, weights, types, bounds)
    rank = rank_preferences(pref, reverse = False)
    rank_results['SPOTIS'] = rank


    # EDAS 
    # EDAS preferences must be sorted in descending order
    edas = EDAS()
    pref = edas(matrix, weights, types)
    rank = rank_preferences(pref, reverse = True)
    rank_results['EDAS'] = rank


    # MABAC
    # MABAC preferences must be sorted in descending order
    mabac = MABAC(normalization_method = norms.minmax_normalization)
    pref = mabac(matrix, weights, types)
    rank = rank_preferences(pref, reverse = True)
    rank_results['MABAC'] = rank


    # MULTIMOORA
    # MULTIMOORA method returns rank
    multimoora = MULTIMOORA()
    rank = multimoora(matrix, weights, types)
    rank_results['MMOORA'] = rank


    # WASPAS
    # WASPAS preferences must be sorted descending order
    waspas = WASPAS(normalization_method = norms.linear_normalization, lambda_param = 0.5)
    pref = waspas(matrix, weights, types)
    rank = rank_preferences(pref, reverse = True)
    rank_results['WASPAS'] = rank


    rank_results = rank_results.set_index('Ai')
    header = [rank_results.index.name]
    header = header + list(rank_results.columns)
    print(tabulate(rank_results, headers = header, tablefmt='orgtbl'))

    # compromise ranking prepared with the Copeland compromise ranking

    compromise_ranking = compromises.copeland(rank_results)

    rank_results_final = copy.deepcopy(rank_results)
    rank_results_final['Compromise'] = compromise_ranking
    header = [rank_results_final.index.name]
    header = header + list(rank_results_final.columns)
    print(tabulate(rank_results_final, headers = header, tablefmt='github'))

    #=======================================================================================================
    # visualization
    # barplot

    df_plot = copy.deepcopy(rank_results)
    plot_barplot(df_plot, 'MCDA methods')
    
    data = copy.deepcopy(rank_results_final)
    method_types = list(data.columns)

    dict_new_heatmap_rw = Create_dictionary()

    for el in method_types:
        dict_new_heatmap_rw.add(el, [])

    dict_new_heatmap_ws = copy.deepcopy(dict_new_heatmap_rw)
    dict_new_heatmap_pearson = copy.deepcopy(dict_new_heatmap_rw)
    dict_new_heatmap_spearman = copy.deepcopy(dict_new_heatmap_rw)

    # heatmaps for correlations coefficients
    for i, j in [(i, j) for i in method_types[::-1] for j in method_types]:
        dict_new_heatmap_rw[j].append(corrs.weighted_spearman(data[i], data[j]))
        dict_new_heatmap_ws[j].append(corrs.WS_coeff(data[i], data[j]))
        dict_new_heatmap_pearson[j].append(corrs.pearson_coeff(data[i], data[j]))
        dict_new_heatmap_spearman[j].append(corrs.spearman(data[i], data[j]))
        

    df_new_heatmap_rw = pd.DataFrame(dict_new_heatmap_rw, index = method_types[::-1])
    df_new_heatmap_rw.columns = method_types

    df_new_heatmap_ws = pd.DataFrame(dict_new_heatmap_ws, index = method_types[::-1])
    df_new_heatmap_ws.columns = method_types

    df_new_heatmap_pearson = pd.DataFrame(dict_new_heatmap_pearson, index = method_types[::-1])
    df_new_heatmap_pearson.columns = method_types

    df_new_heatmap_spearman = pd.DataFrame(dict_new_heatmap_spearman, index = method_types[::-1])
    df_new_heatmap_spearman.columns = method_types


    # correlation matrix with rw coefficient
    draw_heatmap(df_new_heatmap_rw, r'$r_w$')

    # correlation matrix with WS coefficient
    draw_heatmap(df_new_heatmap_ws, r'$WS$')

    # correlation matrix with Pearson coefficient
    draw_heatmap(df_new_heatmap_pearson, r'$Pearson$')

    # correlation matrix with Spearman coefficient
    draw_heatmap(df_new_heatmap_spearman, r'$r_s$')
    

    #plot radar chart
    matplotlib.rc_file_defaults()
    plot_radar(df_plot)
    
    print('Sensitivity analysis')
    #sensitivity analysis

    # load Input vector with percentage values of weights modification for sensitivity analysis
    # percentages = [0.0, 0.05, 0.2, 0.35, 0.5]
    percentages = np.arange(0.05, 0.5, 0.1)

    #load MCDA name: choose from: TOPSIS, CODAS, VIKOR, SPOTIS, EDAS, MABAC, MULTIMOORA, WASPAS
    mcda_name = 'SPOTIS'
    
    sensitivity_analysis = Sensitivity_analysis_weights()
    for j in [0, 1, 2, 3]:
        data_sens = sensitivity_analysis(matrix, weights, types, percentages, mcda_name, j)

        header = [data_sens.index.name]
        header = header + list(data_sens.columns)
        print('Sensitivity analysis for C' + str(j + 1))
        print(tabulate(data_sens, headers = header, tablefmt='github'))
        plot_barplot_sensitivity(data_sens, mcda_name, r'$C_{' + str(j + 1) + '}$')

        #plot
        plot_lineplot_sensitivity(data_sens, mcda_name, r'$C_{' + str(j + 1) + '}$')
    

if __name__ == "__main__":
    main()

