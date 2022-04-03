import copy
import numpy as np
import pandas as pd
import matplotlib
from tabulate import tabulate

from visualizations import plot_barplot, draw_heatmap, plot_boxplot, plot_lineplot_sensitivity, plot_barplot_sensitivity, plot_radar, plot_boxplot_simulation
from pyrepo_mcda.mcda_methods import CODAS, TOPSIS, WASPAS, VIKOR, SPOTIS, EDAS, MABAC, MULTIMOORA
from pyrepo_mcda.mcda_methods import MULTIMOORA_RS as MOORA
from pyrepo_mcda import distance_metrics as dists
from pyrepo_mcda import correlations as corrs
from pyrepo_mcda import normalizations as norms
from pyrepo_mcda import weighting_methods as mcda_weights
from pyrepo_mcda import compromise_rankings as compromises
from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda.sensitivity_analysis_weights_percentages import Sensitivity_analysis_weights_percentages
from pyrepo_mcda.sensitivity_analysis_weights_values import Sensitivity_analysis_weights_values


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
    data = pd.read_csv('data' + '.csv', index_col = 'Ai')
    df_data = data.iloc[:len(data) - 2, :]
    weights = data.iloc[len(data) - 2, :].to_numpy()
    types = data.iloc[len(data) - 1, :].to_numpy()

    matrix = df_data.to_numpy()

    header = [df_data.index.name]
    header = header + list(df_data.columns)
    print(tabulate(df_data, headers = header, tablefmt='github'))
    
    print(types)

    list_alt_names = [r'$A_{' + str(i) + '}$' for i in range(1, df_data.shape[0] + 1)]
    list_crit_names = [r'$C_{' + str(i) + '}$' for i in range(1, df_data.shape[1] + 1)]
    
    matrix = df_data.to_numpy()

    print(weights)

    # Distance metrics used with TOPSIS
    # Exploration of results for different chosen distance metrics
    distance_metrics = [
        dists.euclidean,
        dists.manhattan,
        dists.hausdorff,
        dists.chebyshev,
        dists.bray_curtis,
        dists.canberra,
        dists.lorentzian,
        dists.jaccard
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
    # plot box chart of alternatives preference values
    plot_barplot(df_rankings, legend_title = 'Distance metrics')

    # plot box chart of alternatives preference values
    plot_boxplot(df_preferences.T, 'TOPSIS, preference values distribution')

    # Exploration of different MCDA methods
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
    rank_results_final.to_csv('./results/' + 'results.csv')

    
    # visualization of rankings provided by different MCDA methods
    # barplot
    df_plot = copy.deepcopy(rank_results)
    plot_barplot(df_plot, legend_title='MCDA methods')
    
    
    # correlations heatmaps
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
    
    matplotlib.rc_file_defaults()
    print('Sensitivity analysis')
    # Sensitivity analysis - percentage modification of chosen criterion weight
    
    
    # load input vector with percentage values of chosen criterion weights modification for sensitivity analysis
    # percentages = np.arange(0.25, 0.55, 0.1)
    percentages = np.arange(0.05, 0.55, 0.1)
    

    # Create the object of chosen MCDA method
    method = TOPSIS(normalization_method=norms.minmax_normalization, distance_metric=dists.euclidean)
    
    # Create the sensitivity analysis method object
    sensitivity_analysis = Sensitivity_analysis_weights_percentages()
    
    # Perform sensitivity analysis with weights percentage modification for chosen criteria
    for j in [0, 1, 2, 3]:
        df_sens = sensitivity_analysis(matrix, weights, types, percentages, method, j, [-1, 1])

        header = [df_sens.index.name]
        header = header + list(df_sens.columns)
        print('Sensitivity analysis for C' + str(j + 1))
        print(tabulate(df_sens, headers = header, tablefmt='github'))

        plot_lineplot_sensitivity(df_sens, method.__class__.__name__, list_crit_names[j], "Weight modification in %", "percentage")
        # plot_barplot_sensitivity(df_sens, method.__class__.__name__, list_crit_names[j], "weight_percentage_bar")
    
    # Perform sensitivity analysis with setting chosen weight value to selected criterion
    # other criteria have equal weight values and all criteria weights sum to 1
    sensitivity_analysis_weights_values = Sensitivity_analysis_weights_values()
    weight_values = np.arange(0.05, 0.95, 0.1)
    for j in [0, 1, 2, 3]:
        df_sens = sensitivity_analysis_weights_values(matrix, weight_values, types, method, j)
        header = [df_sens.index.name]
        header = header + list(df_sens.columns)
        print('Sensitivity analysis for C' + str(j + 1))
        print(tabulate(df_sens, headers = header, tablefmt='github'))
        plot_lineplot_sensitivity(df_sens, method.__class__.__name__, list_crit_names[j], "Weight value", "value")
    # example of bar chart usage for sensitivity analysis
    plot_barplot_sensitivity(df_sens, method.__class__.__name__, list_crit_names[j], "weight_values_bar")
    # example of radar chart usage for sensitivity analysis
    plot_radar(df_sens, list_crit_names[j] + ' weight modification', j)


    # Robustness analysis
    # Determining intervals of alternatives performance values for particular positions in rankings
    # Create object of chosen MCDA method
    topsis = TOPSIS(normalization_method=norms.minmax_normalization, distance_metric=dists.euclidean)

    # Create minimum bounds of criteria performance
    bounds_min = np.amin(matrix, axis = 0)
    # Create maximum bounds of criteria performance
    bounds_max = np.amax(matrix, axis = 0)
    bounds = np.vstack((bounds_min, bounds_max))

    # Create ideal Solution `isp`
    isp = np.zeros(matrix.shape[1])
    isp[types == 1] = bounds[1, types == 1]
    isp[types == -1] = bounds[0, types == -1]

    # Create anti-Ideal Solution `asp`
    asp = np.zeros(matrix.shape[1])
    asp[types == 1] = bounds[0, types == 1]
    asp[types == -1] = bounds[1, types == -1]

    # Create dictionary with values of stepwise particular criteria performance change
    indexes = {
        0 : 1,
        1 : 10,
        2 : 5,
        3 : 0.1
    }
    
    # Iterate by all criteria
    for j in range(matrix.shape[1]):
        change_val = indexes[j]
        # dictionary for collecting ranking values after modification of performance values
        dict_results_sim = {
            'Rank' : [],
            'Performance' : [],
            'Alternative' : []
            }
        # Iterate by all Alternatives
        for i in range(matrix.shape[0]):
            vec = np.arange(asp[j], isp[j] + types[j] * change_val, types[j] * change_val)
            # Iterate by all performance values of chosen criterion
            for v in vec:
                new_matrix = copy.deepcopy(matrix)
                new_matrix[i, j] = v
                pref = topsis(new_matrix, weights, types)
                rank = rank_preferences(pref, reverse = True)
                dict_results_sim['Rank'].append(rank[i])
                dict_results_sim['Performance'].append(v)
                dict_results_sim['Alternative'].append(list_alt_names[i])

        df_results_sim = pd.DataFrame(dict_results_sim)
        df_results_sim.to_csv('results/' + 'robustness_C' + str(j + 1) + '.csv')

        plot_boxplot_simulation(df_results_sim, 'Alternative', 'Performance', 'Rank' , 'Alternative', 'Performance', 'TOPSIS, Criterion ' + list_crit_names[j] + ' performance change', 'robustness_C' + str(j + 1))
    

if __name__ == "__main__":
    main()