import numpy as np
import os
import prlqr
import pickle
import glob
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd


def autolabel(bar_plot, text_offset=1.5, num=16, idx=0):
    for idx, rect in enumerate(bar_plot.patches[idx*num:(idx+1)*num:1]):
        height = 20
        if np.isnan(rect.get_height()) or rect.get_height() <= 100.:
            bar_plot.text(rect.get_x() + rect.get_width() * (text_offset / 4 ), height,
                             "{:.1f}".format(rect.get_height()),
                             fontsize=8,
                             ha='center', va='bottom', rotation=90)

def load_results(settings):

    system_name = settings['system'].__class__.__name__
    name = settings['name']

    module_path = os.path.dirname(prlqr.__file__)

    result_path = module_path + '/../results/'
    path = result_path + '/' + system_name + '/' + name + '/'

    results = list()

    for file in glob.glob(path + "*.pickle"):
        with open(file, 'rb') as handle:
            results.append(pickle.load(handle))


    trajectories = settings['training_data']['size']['trajectories']
    Ks = ['K_pr', 'K_nom', 'K_rob', 'K_opt']

    costs_list = list()
    for result in results:

        for K_name in Ks:
            costs = {
                'K_name': np.NaN,
                'experiment_id': np.NaN,
                'n_trajectories': np.NaN,
                'successful': np.NaN,
                'stable': np.NaN,
                'all_successful': np.NaN,
                'all_stable': np.NaN,
                'cost': np.NaN
            }

            costs['K_name'] = K_name
            costs['experiment_id'] = result['experiment_id']
            costs['n_trajectories'] = result['n_trajectories']
            costs['successful'] = result['controller'][K_name] is not None

            cost = result['cost'][K_name + '_cost']

            costs['stable'] = cost < np.Inf
            costs['all_successful'] = result['cost']['all_successful']
            costs['all_stable'] = result['cost']['all_stable'] & result['cost']['all_successful']

            # if cost < 100:
            #     costs['cost'] = cost
            costs['cost'] = cost
            costs_list.append(costs)

    costs = pd.DataFrame(data=costs_list)

    return costs


def plot_results(settings, results):

    dpi=80
    fig_size = (6, 4)
    from matplotlib import gridspec

    sns.set_theme()
    sns.set_context("paper")
    sns.set(rc={'figure.figsize': fig_size,
                "font.size": 8,
                "axes.titlesize": 8,
                "axes.labelsize": 12})

    sns.set_style("whitegrid", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })

    # idx = results['n_trajectories'] <= 80
    # results = results[idx]

    stable_result_idx = results['all_stable'] == True
    stable_results = results

    fig = plt.figure(figsize=fig_size, dpi=dpi)
    gs = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[8, 2])
    ax_cost = fig.add_subplot(gs[0, 0])
    ax_unstable = fig.add_subplot(gs[1, 0], sharex=ax_cost)
    #ax_unsuccessful = fig.add_subplot(gs[2, 0], sharex=ax_cost)

    results = results.assign(Synthesis=results.K_name.map({'K_rob': 'R', 'K_pr': 'PR', 'K_nom': 'CE', 'K_opt': 'T'}))
    stable_results = results.assign(Synthesis=stable_results.K_name.map({'K_rob': 'R', 'K_pr': 'PR', 'K_nom': 'CE', 'K_opt': 'T'}))

    hue_order = ['PR', 'R', 'CE', 'T']

    order = settings['training_data']['size']['trajectories']

    ax = sns.boxplot(x="n_trajectories", hue='Synthesis', y="cost", data=stable_results, linewidth=1,
                     order=order, hue_order=hue_order,
                     showfliers=True,
                     ax=ax_cost)

    ax = sns.stripplot(x="n_trajectories", hue='Synthesis', y="cost", data=stable_results, linewidth=1,
                       order=order, hue_order=hue_order,
                       ax=ax_cost, dodge=True)


    ax.set_yscale("log")

    import matplotlib.ticker as ticker

    ax_cost.yaxis.set_major_locator(ticker.LogLocator(subs='all'))
    ax_cost.grid(False, which='major', axis='x')
    ax.set_xticklabels([])
    ax.set_xlabel(None)

    ax.get_xaxis().set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(b=True, which='minor', linewidth=3.)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[0:4], labels[0:4], title='Synthesis')

    unstable = results
    unstable['help'] = 100.01
    count = settings['runs']


    # ax_unstable = sns.barplot(x="n_trajectories", hue='Synthesis', y="help", data=unstable,
    #                           order=settings['training_data']['size']['trajectories'],
    #                           hue_order=hue_order,
    #                           ax=ax_unstable)



    #ax_unstable.set_ylim(None, 140)
    sns.set_context(rc={'patch.linewidth': 0.0})
    unstable = results
    unstable['help'] = 100.01
    sns.set_palette("pastel")
    # ax_unsuccessful = sns.barplot(x="n_trajectories", hue='Synthesis', y="help", data=unstable,
    #                           order=settings['training_data']['size']['trajectories'],
    #                           hue_order=hue_order,
    #                           ax=ax_unsuccessful)

    sns.set_palette("muted")
    ax_unsuccessful = sns.barplot(x="n_trajectories", hue='Synthesis', y="successful", data=unstable,
                                  order=order,
                                  hue_order=hue_order,
                                  ci=None,
                                  estimator=lambda x: np.sum(x) / count * 100,
                                  ax=ax_unstable)


    # sns.set_palette("muted")
    # ax_unstable = sns.barplot(x="n_trajectories", hue='Synthesis', y="stable", data=unstable,
    #                           order=order,
    #                           hue_order=hue_order,
    #                           ci=None,
    #                           estimator=lambda x: np.sum(x) / count * 100,
    #                           ax=ax_unstable)

    n_bars = len(order) * len(hue_order)
    autolabel(ax_unsuccessful, 2, n_bars, 0)
    #autolabel(ax_unstable, 1, n_bars, 1)

    #autolabel(ax_unsuccessful)

    ax_unstable.set_ylim(None, 140)
    ax_unstable.legend([],[], frameon=False)

    ax_unstable.get_xaxis().set_minor_locator(ticker.AutoMinorLocator(2))
    ax_unstable.grid(b=True, which='minor', linewidth=3.)
    ax_unstable.set_axisbelow(True)

    ax_cost.set(ylabel='Quadratic cost')
    ax_unstable.set(ylabel='% feasible', xlabel='Rollouts')

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

    plt.savefig(path + 'results_gp_' + settings['name'] + '.pdf', bbox_inches='tight', pad_inches=0, dpi=dpi)

    plt.show()



if __name__ == "__main__":

    from prlqr.experiment.definitions.furuta_experiment_paper import settings

    results = load_results(settings)
    plot_results(settings, results)

    from prlqr.experiment.definitions.dean_experiment_paper import settings

    results = load_results(settings)
    plot_results(settings, results)

