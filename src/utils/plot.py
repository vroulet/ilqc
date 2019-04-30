import math
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Computer Modern Roman']})
params = {'axes.labelsize': 14,
          'axes.labelweight': 'bold',
          'font.size': 12,
          'legend.fontsize': 18,
          'xtick.labelsize': 14,
          'ytick.labelsize': 14,
          'lines.linewidth': 3,
          'text.usetex': False,
          'figure.figsize': (8, 6)}
plt.rcParams.update(params)


sns.set_style("ticks")
# sns.set_context("paper", rc={'lines.linewidth': 3, "legend.fontsize": 18, "font.size": 18})

nice_writing = {'iteration': 'Iterations',
                'iter_time': 'Time',
                'func_val': 'Function values',
                'norm_grad': 'Gradient norm',
                'prox_lin': 'RegILQR',
                'acc_prox_lin': 'AccRegILQR',
                'gauss_newton': 'ILQR',
                'two_links_arm': 'Two Links Arm',
                'inverse_pendulum': 'Inverse Pendulum'
                }


def plot_exp(info_exp, x_axis, y_axis, hue='algo', log_scale=True, normalize=True,
             plot_min_until_now=False, ax=None):
    if plot_min_until_now:
        take_min_until_now(hue, info_exp, y_axis)
    if normalize:
        info_exp[y_axis] = info_exp[y_axis]/info_exp[y_axis][0]

    if hue is not None:
        exp_plot = sns.lineplot(
            x=x_axis, y=y_axis, hue=hue, data=info_exp, ax=ax)
    else:
        exp_plot = sns.lineplot(x=x_axis, y=y_axis, data=info_exp, ax=ax)

    if log_scale:
        exp_plot.set(yscale="log")

    exp_plot.set(xlabel=nice_writing[x_axis], ylabel=nice_writing[y_axis])

    exp_plot.legend().set_title('')
    handles, labels = exp_plot.get_legend_handles_labels()
    handles = handles[1:]
    labels = labels[1:]
    labels = [nice_writing[label] for label in labels]
    exp_plot.legend(handles=handles, labels=labels)
    fig = exp_plot.get_figure()
    return fig


def take_min_until_now(hue, info_exp, y_axis):
    for algo in list(info_exp[hue].unique()):
        aux = info_exp.loc[info_exp[hue] == algo, y_axis].values.tolist()

        min_until_now = math.inf
        for i, y_val in enumerate(aux):
            min_until_now = min(min_until_now, deepcopy(y_val))
            aux[i] = min_until_now
        info_exp.loc[info_exp[hue] == algo, y_axis] = aux
