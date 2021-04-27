import numpy as np
import os
import prlqr
import pickle
import glob
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd


def flatten_column(df, column_name):
    repeat_lens = [len(item) if item is not np.nan else 1 for item in df[column_name]]
    df_columns = list(df.columns)
    df_columns.remove(column_name)
    expanded_df = pd.DataFrame(np.repeat(df.drop(column_name, axis=1).values, repeat_lens, axis=0), columns=df_columns)
    flat_column_values = np.hstack(df[column_name].values)
    expanded_df[column_name] = flat_column_values
    expanded_df[column_name].replace('nan', np.nan, inplace=True)
    return expanded_df

def load_results(settings):

    system_name = settings['system'].__class__.__name__
    name = settings['name']

    module_path = os.path.dirname(prlqr.__file__)

    result_path = module_path + '/../results/'
    path = result_path + '/' + system_name + '/' + name + '/'

    print(path)
    results = list()

    for file in glob.glob(path + "*.pickle"):
        with open(file, 'rb') as handle:
            results.append(pickle.load(handle))

    print(len(results))

    scales = settings['scales']
    Ks = ['K_pr', 'K_nom', 'K_rob']



    costs_list = list()
    for result in results:

        for K_name in Ks:

            costs = {
                'K_name': np.NaN,
                'experiment_id': np.NaN,
                'scale': np.NaN,
                'samples': np.NaN,
                'successful': [],
                'cost_stable': [],
                'unstable': [],
                'unstable_freq': []
            }

            costs['K_name'] = K_name
            costs['experiment_id'] = result['experiment_id']
            costs['scale'] = result['scale']
            costs['samples'] = result['cost']['samples']

            if result['controller'][K_name] is None:
                costs['successful'] = False
                costs['cost_stable'] = np.NaN
                costs['unstable'] = np.NaN
                costs['unstable_freq'] = np.NaN
            else:
                costs['successful'] = False
                cost = result['cost'][K_name + '_mean_cost']
                cost_stable = cost[cost < np.Inf]
                count_unstable = np.sum(cost >= np.Inf)

                costs['cost_stable'] = cost_stable
                costs['unstable'] = count_unstable
                costs['unstable_freq'] = count_unstable / len(cost) * 100

            costs_list.append(costs)


    costs = pd.DataFrame(data=costs_list)

    costs = flatten_column(costs, 'cost_stable')

    return costs


def plot_results(settings, results, exp_id=0):

    dpi=80
    fig_size = (12, 5)
    exp = results['experiment_id'] == exp_id
    results = results[exp]

    results = results.assign(Synthesis=results.K_name.map({'K_rob': 'R', 'K_pr': 'PR', 'K_nom': 'CE'}))

    hue_order = ['PR', 'R', 'CE']
    sns.set_theme()
    sns.set_context("paper")
    sns.set(rc={'figure.figsize': fig_size,
                "font.size": 8,
                "axes.titlesize": 8,
                "axes.labelsize": 12})
    sns.set_style("whitegrid")
    sns.set_style("whitegrid", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })


    fig = plt.figure(figsize=fig_size, dpi=dpi)

    from matplotlib import gridspec
    gs = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[9, 1])

    ax_cost = fig.add_subplot(gs[0, 0])

    # This will just plot the meanline :) hacks
    meanlineprops = dict(linestyle='--', linewidth=2.5, color='k')
    whiskerprops = dict(linewidth=0)
    ax = sns.boxplot(x="scale", hue='Synthesis', y="cost_stable", data=results, linewidth=1,
                     order=settings['scales'], hue_order=hue_order,
                     showmeans=True, meanprops=meanlineprops, showfliers=False, meanline=True,
                     ax=ax_cost,
                     showbox=False, showcaps=False, whiskerprops=whiskerprops)

    ax = sns.boxenplot(x="scale", hue='Synthesis', y="cost_stable", data=results, linewidth=1,
                     order=settings['scales'], hue_order=hue_order,
                     showfliers=True,
                     ax=ax_cost)


    mean = results.groupby(['scale', 'Synthesis']).mean().reset_index()

    # sns.pointplot(x='scale', y='cost_stable', hue='Synthesis', data=mean,
    #               order=settings['scales'], hue_order=hue_order,
    #               ax=ax_cost, orient='v', join=False)

    ax.set_yscale("log")
    ax.set_ylim(0.01, 1.)

    import matplotlib.ticker as ticker

    ax_cost.yaxis.set_major_locator(ticker.LogLocator(subs='all'))
    ax_cost.grid(False, which='major', axis='x')
    ax.set_xticklabels([])
    ax.set_xlabel(None)

    ax.get_xaxis().set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(b=True, which='minor', linewidth=3.)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[0:3], labels[0:3], title='Synthesis')
    #ax.legend().set_title('Synthesis')


    unstable = results.drop(columns=['cost_stable']).drop_duplicates()

    unstable['help'] = 100.01

    ax_unstable = fig.add_subplot(gs[1, 0], sharex=ax_cost)

    sns.set_palette("pastel")
    ax_unstable = sns.barplot(x="scale", hue='Synthesis', y="help", data=unstable,
                              order=settings['scales'], hue_order=hue_order,
                              ax=ax_unstable)


    sns.set_palette("dark")
    ax_unstable = sns.barplot(x="scale", hue='Synthesis', y="unstable_freq", data=unstable,
                              order=settings['scales'], hue_order=hue_order,
                              ax=ax_unstable)

    ax_unstable.set_ylim(None, 140)

    def autolabel(bar_plot):
        for idx, rect in enumerate(bar_plot.patches):
            height = 105
            if np.isnan(rect.get_height()) or rect.get_height() <= 100.:
                bar_plot.text(rect.get_x() + rect.get_width() / 2., height,
                                 "{:.1f}".format(rect.get_height()),
                                 fontsize=8,
                                 ha='center', va='bottom', rotation=90)

    autolabel(ax_unstable)

    ax_unstable.legend([],[], frameon=False)

    ax_unstable.get_xaxis().set_minor_locator(ticker.AutoMinorLocator(2))
    ax_unstable.grid(b=True, which='minor', linewidth=3.)
    ax_unstable.set_axisbelow(True)

    ax_cost.set(ylabel='Quadratic cost')
    ax_unstable.set(ylabel='% unstable', xlabel='Indicator for model uncertainty ($\sigma^2$)')

    from matplotlib.ticker import FuncFormatter
    ax_cost.xaxis.set_major_formatter(FuncFormatter(lambda x, p: '{:1.0e}'.format(settings['scales'][p])))

    sns.despine(ax=ax_unstable, left=True)
    sns.despine(ax=ax_cost, left=True)

    plt.tight_layout()
    plt.margins(0, 0)

    import os
    from pathlib import Path

    module_path = os.path.dirname(prlqr.__file__)
    path = module_path + '/../figures/'

    dir = Path(path)
    dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(path + 'results_' + settings['name'] + '_sample_' + str(i) + '.pdf', bbox_inches='tight', pad_inches=0, dpi=dpi)

    plt.show()

    # sns.catplot(x="scale", hue='K_name', y="help", data=unstable,
    #                  order=settings['scales'], hue_order=['K_pr', 'K_rob', 'K_nom'])
    #
    # plt.show()

    # g = sns.catplot(x="K_name", y="cost_stable", col="scale", kind="box", data=results, col_wrap = 4,
    #                 col_order=settings['scales'][0:8], order=['K_pr', 'K_rob', 'K_nom'])
    # g.set(yscale='log')
    # g.set(ylim = (0.02, 1.))
    # plt.show()

if __name__ == "__main__":
    from prlqr.experiment.definitions.synthesis_dean_experiment_paper import settings
    results = load_results(settings)

    for i in range(5):
        plot_results(settings, results, i)
