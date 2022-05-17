import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# bar (column) chart
def plot_barplot(df_plot, legend_title):
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
    plt.savefig('./results/' + 'bar_chart_' + legend_title + '.eps')
    plt.show()


# bar (column) chart for sensitivity analysis
def plot_barplot_sensitivity(df_plot, method_name, criterion_name, filename = ""):
    """
    Visualization method to display column chart of alternatives rankings obtained with 
    modification of weight of given criterion.

    Parameters
    ----------
        df_plot : DataFrame
            DataFrame containing rankings of alternatives obtained with different weights of
            selected criterion. The particular rankings are contained in subsequent columns of 
            DataFrame.

        method_name : str
            Name of chosen MCDA method, i.e. `TOPSIS`, `VIKOR`, `CODAS`, `WASPAS`, `MULTIMOORA`, `MABAC`, `EDAS`, `SPOTIS`
        
        criterion_name : str
            Name of chosen criterion whose weight is modified

        filename : str
            Name of file to save this chart

    Examples
    -----------
    >>> plot_barplot_sensitivity(df_plot, method_name, criterion_name, filename)
    """
    step = 1
    list_rank = np.arange(1, len(df_plot) + 1, step)

    ax = df_plot.plot(kind='bar', width = 0.8, stacked=False, edgecolor = 'black', figsize = (9,4))
    ax.set_xlabel('Alternatives', fontsize = 12)
    ax.set_ylabel('Rank', fontsize = 12)
    ax.set_yticks(list_rank)

    ax.set_xticklabels(df_plot.index, rotation = 'horizontal')
    ax.tick_params(axis='both', labelsize=12)
    y_ticks = ax.yaxis.get_major_ticks()
    ax.set_ylim(0, len(df_plot) + 1)
    ax.set_title(method_name + ', modification of ' + criterion_name + ' weights')

    plt.legend(bbox_to_anchor=(1.0, 0.82, 0.3, 0.2), loc='upper left', title = 'Weights change', edgecolor = 'black', fontsize = 12)

    ax.grid(True, linestyle = ':')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('./results/' + 'sensitivity_' + 'hist_' + method_name + '_' + criterion_name + '_' + filename + '.eps')
    plt.show()


# plot line chart for sensitivity analysis
def plot_lineplot_sensitivity(data_sens, method_name, criterion_name, x_title, filename = ""):
    """
    Visualization method to display line chart of alternatives rankings obtained with 
    modification of weight of given criterion.

    Parameters
    ----------
        df_plot : DataFrame
            DataFrame containing rankings of alternatives obtained with different weight of
            selected criterion. The particular rankings are contained in subsequent columns of 
            DataFrame.

        method_name : str
            Name of chosen MCDA method, i.e. `TOPSIS`, `VIKOR`, `CODAS`, `WASPAS`, `MULTIMOORA`, `MABAC`, `EDAS`, `SPOTIS`
        
        criterion_name : str
            Name of chosen criterion whose weight is modified
        
        x_title : str
            Title of x axis

        filename : str
            Name of file to save this chart

    Examples
    ----------
    >>> plot_lineplot_sensitivity(df_plot, method_name, criterion_name, x_title, filename)
    """
    plt.figure(figsize = (6, 3))
    for j in range(data_sens.shape[0]):
        
        plt.plot(data_sens.iloc[j, :], linewidth = 2)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        plt.annotate(data_sens.index[j], (x_max, data_sens.iloc[j, -1]),
                        fontsize = 12, style='italic',
                        horizontalalignment='left')

    plt.xlabel(x_title, fontsize = 12)
    plt.ylabel("Rank", fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.title(method_name + ', modification of ' + criterion_name + ' weight')
    plt.grid(True, linestyle = ':')
    plt.tight_layout()
    plt.savefig('./results/' + 'sensitivity_' + 'lineplot_' + method_name + '_' + criterion_name + '_' + filename + '.eps')
    plt.show()


# heat maps with correlations
def draw_heatmap(df_new_heatmap, title):
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
    heatmap = sns.heatmap(df_new_heatmap, annot=True, fmt=".2f", cmap="PuBu",
                          linewidth=0.5, linecolor='w')
    plt.yticks(va="center")
    plt.xlabel('Methods')
    plt.title('Correlation: ' + title)
    plt.tight_layout()
    plt.savefig('./results/' + 'correlations_' + title + '.eps')
    plt.show()


# radar chart
def plot_radar(data, title, j):
    """
    Visualization method to display rankings of alternatives obtained with different methods
    on the radar chart.

    Parameters
    -----------
        data : DataFrame
            DataFrame containing containing rankings of alternatives obtained with different 
            methods. The particular rankings are contained in subsequent columns of DataFrame.

        title : str
            Chart title

        j : int
            Index of criterion chosen for weight modification in sensitivity analysis

    Examples
    ----------
    >>> plot_radar(data, title, j)
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
    ax.grid(True)
    ax.set_axisbelow(True)
    plt.legend(data.columns, bbox_to_anchor=(1.0, 0.95, 0.4, 0.2), loc='upper left')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('./results/' + 'radar_chart_C' + str(j + 1) + '.eps')
    plt.show()

# Examplary visualization method
def plot_boxplot(data, title):
    """
    Display boxplot showing distribution of preference values determined with different methods.

    Parameters
    ----------
        data : dataframe
            dataframe with correlation values between compared rankings

        title : str
            Title of chart.

    Examples
    ----------
    >>> plot_boxplot(data, title)
    """
    
    df_melted = pd.melt(data)
    plt.figure(figsize = (7, 4))
    ax = sns.boxplot(x = 'variable', y = 'value', data = df_melted, width = 0.4)
    ax.grid(True, linestyle = '--')
    ax.set_axisbelow(True)
    ax.set_xlabel('Alternatives', fontsize = 12)
    ax.set_ylabel('Preference distribution', fontsize = 12)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('./results/' + 'TOPSIS_boxplot' + '.eps')
    plt.show()


# plot box chart of results obtained in robustness analysis showing intervals of alternatives performance values for particular ranks
def plot_boxplot_simulation(data, x, y, hue, xtitle, ytitle, title, filename):
    """
    Visualization method to display box chart of results obtained in robustness analysis. It shows
    intervals of alternatives performance values for particular positions in ranking.

    Parameters
    ----------
        data : DataFrame
            DataFrame containing results

        x : str
            Name of column in DataFrame with variable names in axis x on chart

        y : str
            Name of column in DataFrame with variable values in axis y on chart
        hue : str
            Name of hue, that determines how the data are plotted

        xtitle : str
            Name of axis x title

        ytitle : str
            Name of axis y title

        title : str
            Chart title

        filename : str
            Name of file in which chart will be saved

    Examples
    ----------
    >>> plot_boxplot_simulation(data, x, y, hue , xtitle, ytitle, title, filename)

    """
    plt.figure(figsize = (9,5))
    ax = sns.boxplot(x = x, y = y, hue = hue, palette='husl', data = data)
    ax.set_xlabel(xtitle, fontsize = 12)
    ax.set_ylabel(ytitle, fontsize = 12)
    ax.grid(True, linestyle = ':')
    ax.set_axisbelow(True)
    plt.legend(bbox_to_anchor=(1.0, 0.82, 0.3, 0.2), loc='upper left', title = 'Rank', edgecolor = 'black', fontsize = 12)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('./results/' + filename + '.eps')
    plt.show()