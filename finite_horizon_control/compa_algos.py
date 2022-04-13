import sys
import torch
from pandas import DataFrame
from matplotlib import pyplot as plt
import seaborn as sns
from copy import deepcopy

sys.path.append('..')
sys.path.append('.')

from finite_horizon_control.fhc_pipeline import solve_ctrl_pb, solve_ctrl_pb_incrementally
from finite_horizon_control.utils_plot import nice_ax, get_nice_writing, set_plt_params, \
                           get_palette_line_styles,  get_var_palette_line_styles, format_plot

torch.set_default_tensor_type(torch.DoubleTensor)


def compa_algos(env_cfg, optim_cfgs, x_axis, y_axis, scale=True,
                plot=False, logscale=False, keep_results=True):
    compa_optim = DataFrame()
    for optim_cfg in optim_cfgs:

        if keep_results:
            exp_outputs = solve_ctrl_pb_incrementally(env_cfg, optim_cfg)
        else:
            exp_outputs = solve_ctrl_pb(env_cfg, optim_cfg)
        conv_algo = exp_outputs['metrics']

        compa_optim = compa_optim.append(DataFrame(conv_algo), ignore_index=True)

    if scale:
        min_info = min(compa_optim[y_axis])
        aux = [(info - min_info + 1e-12) / (compa_optim[y_axis].to_list()[0] - min_info + 1e-12)
               for info in compa_optim[y_axis]]
        compa_optim[y_axis] = aux
        compa_optim[compa_optim[y_axis] <= 1e-12] = float('nan')
    compa_optim = compa_optim.append(compa_optim, ignore_index=True)

    if x_axis == 'time':
        mean_final_time = compa_optim.groupby('algo')['time'].max().mean()
        compa_optim = compa_optim[compa_optim['time'] <= mean_final_time]

    if plot:
        set_plt_params()
        fig = plt.figure()
        ax = sns.lineplot(x=x_axis, y=y_axis, hue='algo', data=compa_optim)
        handles, labels = nice_ax(ax, x_axis, y_axis, logscale)
        fig.legend(handles=handles, labels=labels, loc='center', bbox_to_anchor=(0.5, 1.), ncol=len(optim_cfgs))
        plt.tight_layout()
        plt.show()
    return compa_optim


def compa_conv_algos_envs(env_cfgs, algos, horizons, max_iter, x_axis, approx, var_plot_style=False):
    # plots_folder = '/Users/vincentroulet/Dropbox/Postdoc/ilqc/tech_report/fig'
    nice_writing = get_nice_writing()
    set_plt_params()
    palette, marker_styles = get_palette_line_styles() if not var_plot_style else get_var_palette_line_styles()
    start_iter_plot, max_iter_plot = 0, max_iter - 5

    n_rows, n_cols = len(env_cfgs), len(horizons)
    fig, big_axes = plt.subplots(figsize=(20, 24), nrows=n_rows, ncols=1, sharey=True)
    for row, big_ax in enumerate(big_axes, start=1):
        env_cfg = env_cfgs[row-1]
        big_ax.set_title(nice_writing[env_cfg['env']], fontsize=36, y=1.15)
        big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
        big_ax._frameon = False

    for i, env_cfg in enumerate(env_cfgs):
        algos_compared = deepcopy(algos)

        total_time = 1. if env_cfg['env'] == 'real_car' else 2.
        if env_cfg['env'] == 'real_car' and 'gd' in algos:
            algos_compared.remove('gd')

        for j, horizon in enumerate(horizons):
            ax = fig.add_subplot(n_rows, n_cols, i*n_cols + j + 1)
            env_cfg.update(dt=total_time / horizon, horizon=horizon)

            optim_cfgs = [dict(max_iter=max_iter, algo=algo) for algo in algos_compared]
            compa_optim = compa_algos(env_cfg, optim_cfgs, x_axis, 'cost')

            compa_optim = compa_optim[compa_optim['iteration']>=start_iter_plot]
            compa_optim = compa_optim[compa_optim['iteration']<=max_iter_plot]

            sns.lineplot(x=x_axis, y='cost', hue='algo', style='algo',  data=compa_optim, ax=ax,
                         markers=marker_styles, dashes=False, palette=palette, markevery=5)
            ax.set_title(rf'$\tau={horizon}$')
            ax.set(yscale='log')

    format_plot(fig, x_axis, 'cost', n_rows, n_cols)
    plt.show()

    # fig.savefig(os.path.join(plots_folder, '_'.join([approx, x_axis, 'cost']) + '.pdf'),
    #             bbox_inches='tight', format='pdf')


def stepsize_analysis(env_cfgs, algotype_approxs, horizons, max_iter, approx, var_plot_style=False):
    assert 'gd' not in algotype_approxs
    # plots_folder = '/Users/vincentroulet/Dropbox/Postdoc/ilqc/tech_report/fig'
    nice_writing = get_nice_writing()
    set_plt_params()
    palette, marker_styles = get_palette_line_styles() if not var_plot_style else get_var_palette_line_styles()

    stepsize_modes = ['dir', 'reg']
    n_rows_fig, n_rows_per_fig = len(env_cfgs), len(stepsize_modes)
    n_rows, n_cols = n_rows_fig*n_rows_per_fig, len(horizons)
    fig, big_axes = plt.subplots(figsize=(20, 25), nrows=n_rows, ncols=1, sharey=True)

    for row, big_ax in enumerate(big_axes, start=1):
        if row % len(stepsize_modes) == 1:
            env_cfg = env_cfgs[int((row-1)/n_rows_per_fig)]
            big_ax.set_title(nice_writing[env_cfg['env']], fontsize=36, y=1.2)
        big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
        big_ax._frameon = False

    for i, env_cfg in enumerate(env_cfgs):
        total_time = 1. if env_cfg['env'] == 'real_car' else 2.

        for k, stepsize_mode in enumerate(stepsize_modes):
            algos = [algo_type_approx + '_' + stepsize_mode for algo_type_approx in algotype_approxs]
            logscale = stepsize_mode == 'reg'
            for j, horizon in enumerate(horizons):
                ax = fig.add_subplot(n_rows, n_cols, i*n_cols*n_rows_per_fig + k*n_cols + j + 1)
                dt = total_time / horizon
                env_cfg.update(dt=dt)

                optim_cfgs = [dict(max_iter=max_iter, algo=algo, horizon=horizon) for algo in algos]

                compa_optim = compa_algos(env_cfg, optim_cfgs, 'iteration', 'stepsize', scale=False)

                ax = sns.lineplot(x='iteration', y='stepsize', hue='algo', style='algo', data=compa_optim, ax=ax,
                                  markers=marker_styles, dashes=False, palette=palette, markevery=5)
                ax.set_title(rf'$\tau={horizon}$')
                if logscale:
                    ax.set(yscale='log')
    format_plot(fig, 'iteration', 'stepsize', n_rows, n_cols)
    plt.show()

    # fig.savefig(os.path.join(plots_folder, '_'.join([approx, 'stepsize']) + '.pdf'),
    #             bbox_inches='tight', format='pdf')


def plot_compa_algos_envs(approx):
    algos = ['gd'] if approx == 'linquad' else []
    algos = algos + [algo_type + '_' + approx + '_' + step_mode
                     for algo_type in ['classic', 'ddp'] for step_mode in ['reg', 'dir']]
    env_cfgs = [
        dict(env='pendulum'),
        dict(env='simple_car', track='simple', cost='exact', discretization='euler', reg_bar=0.),
        dict(env='real_car', track='simple', cost='contouring', discretization='rk4_cst_ctrl')
    ]
    max_iter = 50
    horizons = [25, 50, 100]
    for x_axis in ['iteration', 'time']:
        compa_conv_algos_envs(env_cfgs, algos, horizons, max_iter, x_axis, approx)


def plot_stepsize_behavior(approx):
    algotype_approxs = [algo_type + '_' + approx for algo_type in ['classic', 'ddp']]
    env_cfgs = [
        dict(env='pendulum'),
        dict(env='simple_car', track='simple', cost='exact', discretization='euler', reg_bar=0.),
    ]

    max_iter = 50
    horizons = [25, 50, 100]
    stepsize_analysis(env_cfgs, algotype_approxs, horizons, max_iter, approx)


def plot_stepsize_strategies(algo_type, approx):
    algos = [algo_type + '_' + approx + '_' + stepsize_mode
             for stepsize_mode in ['dir', 'dirvar', 'reg', 'regvar']]
    env_cfgs = [
        dict(env='pendulum'),
        dict(env='simple_car', track='simple', cost='exact', discretization='euler', reg_bar=0.),
        dict(env='real_car', track='simple', cost='contouring', discretization='rk4_cst_ctrl')
    ]

    max_iter = 50
    horizons = [25, 50, 100]
    for x_axis in ['iteration', 'time']:
        compa_conv_algos_envs(env_cfgs, algos, horizons, max_iter, x_axis,
                         f'stepsize_strat_{algo_type}_{approx}', var_plot_style=True)


if __name__ == '__main__':
    plot_compa_algos_envs('linquad')
    # plot_stepsize_behavior('linquad')
    plot_compa_algos_envs('quad')
    # plot_stepsize_behavior('quad')
    # plot_stepsize_strategies('classic', 'linquad')

