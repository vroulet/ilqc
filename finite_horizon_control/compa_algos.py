import sys
import torch
import os
from pandas import DataFrame
from matplotlib import pyplot as plt
import seaborn as sns
from copy import deepcopy

sys.path.append('..')
sys.path.append('.')

from finite_horizon_control.fhc_pipeline import solve_ctrl_pb_incrementally
from finite_horizon_control.utils_plot import nice_ax, get_nice_writing, set_plt_params, \
                           get_palette_line_styles, format_plot

torch.set_default_tensor_type(torch.DoubleTensor)


def compa_conv_algos(env_cfg, optim_cfgs, x_axis, y_axis, plot=False, logscale=False):
    compa_optim = DataFrame()
    for optim_cfg in optim_cfgs:
        exp_outputs = solve_ctrl_pb_incrementally(env_cfg, optim_cfg)
        conv_algo = exp_outputs['metrics']
        compa_optim = compa_optim.append(DataFrame(conv_algo), ignore_index=True)

    if y_axis == 'cost':
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


total_times = dict(pendulum=2., cart_pendulum=2.5, simple_car=2., real_car=1.)


def compa_algos_horizons(env_cfg, algos, horizons, max_iter, x_axis, y_axis, add_legend=True):
    set_plt_params()
    nice_writing = get_nice_writing()
    palette, marker_styles = get_palette_line_styles()
    start_iter_plot, max_iter_plot = 0, max_iter - 5

    fig, axs = plt.subplots(1, len(horizons), squeeze=False, figsize=(18, 5.5))
    total_time = total_times[env_cfg['env']]

    algos_compared = deepcopy(algos)
    if env_cfg['env'] == 'real_car' and 'gd' in algos:
        algos_compared.remove('gd')

    for j, horizon in enumerate(horizons):
        env_cfg.update(dt=total_time / horizon, horizon=horizon)

        optim_cfgs = [dict(max_iter=max_iter, algo=algo) for algo in algos_compared]
        compa_optim = compa_conv_algos(env_cfg, optim_cfgs, x_axis, y_axis)

        compa_optim = compa_optim[compa_optim['iteration']>=start_iter_plot]
        compa_optim = compa_optim[compa_optim['iteration']<=max_iter_plot]

        sns.lineplot(x=x_axis, y=y_axis, hue='algo', style='algo',  data=compa_optim, ax=axs[0][j],
                     markers=marker_styles, dashes=False, palette=palette, markevery=5)
        axs[0][j].set_title(rf'$\tau={horizon}$')
        axs[0][j].set(yscale='log')

    format_plot(fig, add_legend)
    fig.suptitle(nice_writing[env_cfg['env']], y=1.)
    return fig


if __name__ == '__main__':
    plots_folder = os.path.dirname(os.path.abspath(__file__))

    env_cfgs = [
        dict(env='pendulum'),
        dict(env='cart_pendulum', x_limits=(-2., 2.), stay_put_time=0.6),
        dict(env='simple_car', track='simple', cost='exact', discretization='euler', reg_bar=0.),
        dict(env='real_car', track='simple', cost='contouring', discretization='rk4_cst_ctrl')
    ]
    max_iter = 50
    horizons = [25, 50, 100]

    y_axis = 'cost'
    for env_cfg in env_cfgs:
        for approx in ['linquad', 'quad']:
            algos = ['gd'] + [algo_type + '_' + approx + '_' + step_mode
                              for algo_type in ['classic', 'ddp'] for step_mode in ['reg', 'dir']]
            for x_axis in ['iteration', 'time']:
                fig = compa_algos_horizons(env_cfg, algos, horizons, max_iter, x_axis, y_axis)
                plt.show()

    x_axis = 'iteration'
    y_axis = 'stepsize'
    for env_cfg in env_cfgs:
        for approx in ['linquad', 'quad']:
            for step_mode in ['reg', 'dir']:
                algos = [algo_type + '_' + approx + '_' + step_mode
                         for algo_type in ['classic', 'ddp']]
                fig = compa_algos_horizons(env_cfg, algos, horizons, max_iter, x_axis, y_axis)
                plt.show()
