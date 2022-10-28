import sys
import torch
import os
import pandas as pd
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


def compa_algos_env(env_cfg, optim_cfgs, x_axis, y_axis, plot=False, logscale=False, normalize=True):
    compa_optim = DataFrame()
    for optim_cfg in optim_cfgs:
        exp_outputs = solve_ctrl_pb_incrementally(env_cfg, optim_cfg)
        conv_algo = exp_outputs['metrics']
        compa_optim = compa_optim.append(DataFrame(conv_algo), ignore_index=True)

    if y_axis == 'cost':
        min_info = min(compa_optim[y_axis])
        if normalize:
            aux = [(info - min_info + 1e-12) / (compa_optim[y_axis].to_list()[0] - min_info + 1e-12)
                   for info in compa_optim[y_axis]]
        else:
            aux = [info / compa_optim[y_axis].to_list()[0] for info in compa_optim[y_axis]]
        # if env_cfg['env'] == 'cart_pendulum' or env_cfg['env'] == 'simple_car':
        #     aux = [max(info, 1e-6) for info in aux]
        compa_optim[y_axis] = aux
    # pd.set_option('display.expand_frame_repr', False, 'display.max_rows', None, 'display.max_columns', None)
    # print(compa_optim)

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


def compa_algos_env_horizons(env_cfg, algos, horizons, max_iter, x_axis, y_axis, add_legend=True):
    set_plt_params()
    nice_writing = get_nice_writing()
    palette, marker_styles = get_palette_line_styles()
    start_iter_plot, max_iter_plot = 0, max_iter - 5

    fig, axs = plt.subplots(1, len(horizons), squeeze=False, figsize=(16, 5.5))
    total_time = total_times[env_cfg['env']]

    algos_compared = deepcopy(algos)
    if env_cfg['env'] == 'real_car' and 'gd' in algos:
        algos_compared.remove('gd')

    for j, horizon in enumerate(horizons):
        env_cfg.update(dt=total_time / horizon, horizon=horizon)

        optim_cfgs = [dict(max_iter=max_iter, algo=algo) for algo in algos_compared]
        compa_optim = compa_algos_env(env_cfg, optim_cfgs, x_axis, y_axis)

        compa_optim = compa_optim[compa_optim['iteration']>=start_iter_plot]
        compa_optim = compa_optim[compa_optim['iteration']<=max_iter_plot]

        sns.lineplot(x=x_axis, y=y_axis, hue='algo', style='algo',  data=compa_optim, ax=axs[0][j],
                     markers=marker_styles, dashes=False, palette=palette, markevery=10)
        if axs[0][j].get_legend() is not None:
            axs[0][j].get_legend().remove()
        x_axis = axs[0][j].xaxis.get_label().get_text()
        axs[0][j].set_xlabel(nice_writing[x_axis])
        if j == 0:
            y_axis = axs[0][j].yaxis.get_label().get_text()
            axs[0][j].set_ylabel(nice_writing[y_axis])
        else:
            axs[0][j].set(ylabel=None)
        axs[0][j].set(yscale='log')
        axs[0][j].set_title(rf'$\tau={horizon}$')

    fig.suptitle(nice_writing[env_cfg['env']], y=1.)
    fig.tight_layout()
    if add_legend:
        handles, labels = axs[0][0].get_legend_handles_labels()
        labels = [nice_writing[label] for label in labels]
        fig.legend(handles=handles, labels=labels, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=len(handles))

    return fig


def compa_algos_env_stepsizes(env_cfg, algos, horizons, max_iter, x_axis, y_axis, add_legend=True):
    set_plt_params()
    nice_writing = get_nice_writing()
    palette, marker_styles = get_palette_line_styles()
    start_iter_plot, max_iter_plot = 0, max_iter - 5

    fig, axs = plt.subplots(2, len(horizons), squeeze=False, figsize=(16, 8))
    total_time = total_times[env_cfg['env']]
    axs_handles = []
    axs_labels = []
    for i, step_mode in enumerate(['dir', 'reg']):
        for j, horizon in enumerate(horizons):
            env_cfg.update(dt=total_time / horizon, horizon=horizon)

            algos_compared = [algo + f'_{step_mode}' for algo in algos]

            optim_cfgs = [dict(max_iter=max_iter, algo=algo) for algo in algos_compared]
            compa_optim = compa_algos_env(env_cfg, optim_cfgs, x_axis, y_axis)

            compa_optim = compa_optim[compa_optim['iteration']>=start_iter_plot]
            compa_optim = compa_optim[compa_optim['iteration']<=max_iter_plot]

            sns.lineplot(x=x_axis, y=y_axis, hue='algo', style='algo',  data=compa_optim, ax=axs[i][j],
                         markers=marker_styles, dashes=False, palette=palette, markevery=5)
            if i == 0:
                axs[i][j].set_title(rf'$\tau={horizon}$')
            axs[i][j].set(yscale='log')
            if i == 1:
                x_axis = axs[i][j].xaxis.get_label().get_text()
                axs[i][j].set_xlabel(nice_writing[x_axis])
            else:
                axs[i][j].set(xlabel=None)
            if j == 0:
                y_axis = axs[i][j].yaxis.get_label().get_text()
                axs[i][j].set_ylabel(nice_writing[y_axis])
            else:
                axs[i][j].set(ylabel=None)
            if axs[i][j].get_legend() is not None:
                axs[i][j].get_legend().remove()

            handles, labels = axs[i][j].get_legend_handles_labels()
            for k, label in enumerate(labels):
                if nice_writing[label] not in axs_labels:
                    axs_handles.append(axs[i][j].lines[k])
                    axs_labels.append(nice_writing[labels[k]])

    fig.suptitle(nice_writing[env_cfg['env']], y=1.)
    fig.tight_layout()
    if add_legend:
        fig.legend(handles=axs_handles, labels=axs_labels, loc='center',
                   bbox_to_anchor=(0.5, 0.02), ncol=len(axs_handles))
    return fig


def plot_conv_all_algos():
    env_cfgs = [
        dict(env='pendulum'),
        dict(env='cart_pendulum', x_limits=(-2., 2.), stay_put_time=0.6),
        dict(env='simple_car', track='simple', cost='exact', discretization='euler', reg_bar=0.),
        dict(env='real_car', track='simple', cost='contouring', discretization='rk4_cst_ctrl')
    ]
    max_iter = 80
    horizons = [25, 50, 100]

    nice_writing = get_nice_writing()
    y_axis = 'cost'
    for approx in ['linquad', 'quad']:
        for x_axis in ['iteration', 'time']:
            for env_cfg in env_cfgs:
                algos = ['gd'] if approx == 'linquad' else []
                algos = algos + [algo_type + '_' + approx + '_' + step_mode
                                 for algo_type in ['classic', 'ddp'] for step_mode in ['reg', 'dir']]

                fig = compa_algos_env_horizons(env_cfg, algos, horizons, max_iter, x_axis, y_axis, add_legend=False)
                if env_cfg['env'] == 'pendulum':
                    handles, labels = fig.axes[0].get_legend_handles_labels()
                    labels = [nice_writing[label] for label in labels]
                if env_cfg['env'] == 'real_car':
                    fig.legend(handles=handles, labels=labels, loc='center', bbox_to_anchor=(0.5, 0.),
                               ncol=len(handles))
                fig.tight_layout()
                fig.show()
                # path = '/Users/vincentroulet/git_projects/vincent/ctrl/papers/techreport_arxiv2022/fig' \
                #         + '/{0}_{1}_{2}_{3}.pdf'.format(env_cfg['env'], approx, x_axis, y_axis)
                # fig.savefig(path, format='pdf', bbox_inches='tight')

    env_cfgs = [
        dict(env='pendulum'),
        dict(env='cart_pendulum', x_limits=(-2., 2.), stay_put_time=0.6),
        dict(env='simple_car', track='simple', cost='exact', discretization='euler', reg_bar=0.),
    ]

    x_axis = 'iteration'
    y_axis = 'stepsize'
    for approx in ['linquad', 'quad']:
        for env_cfg in env_cfgs:
            algos = [algo_type + '_' + approx for algo_type in ['classic', 'ddp']]
            fig = compa_algos_env_stepsizes(env_cfg, algos, horizons, max_iter, x_axis, y_axis, add_legend=False)
            if env_cfg['env'] == 'pendulum':
                axs_handles = []
                axs_labels = []
                for ax in fig.axes:
                    handles, labels = ax.get_legend_handles_labels()
                    for k, label in enumerate(labels):
                        if nice_writing[label] not in axs_labels:
                            axs_handles.append(ax.lines[k])
                            axs_labels.append(nice_writing[labels[k]])

            if env_cfg['env'] == 'simple_car':
                fig.legend(handles=axs_handles, labels=axs_labels, loc='center', bbox_to_anchor=(0.5, 0.),
                           ncol=len(axs_handles))

            fig.tight_layout()
            fig.show()
            # path = '/Users/vincentroulet/git_projects/vincent/ctrl/papers/techreport_arxiv2022/fig' \
            #        + '/{0}_{1}_{2}_{3}.pdf'.format(env_cfg['env'], approx, x_axis, y_axis)
            # fig.savefig(path, format='pdf', bbox_inches='tight')


def plot_conv_gd_ilqr_iddp():
    set_plt_params()

    env_cfgs = [
        dict(env='pendulum', reg_ctrl=0., horizon=100, dt=2./100),
        dict(env='simple_car', track='simple', cost='exact', discretization='rk4', reg_bar=0., reg_ctrl=0.,
             horizon=50, dt=2./50),
    ]
    max_iters = dict(pendulum=30, simple_car=500)
    nice_writing = get_nice_writing(alternative=True)
    palette, marker_styles = get_palette_line_styles()

    x_axis, y_axis = 'iteration', 'cost'
    approx, step_mode = 'linquad', 'reg'
    algos = ['gd'] + [algo_type + '_' + approx + '_' + step_mode for algo_type in ['classic', 'ddp']]

    fig, axs = plt.subplots(1, len(env_cfgs), figsize=(14, 7))
    # fig, axs = plt.subplots(2, 1, figsize=(7, 12))

    for i, env_cfg in enumerate(env_cfgs):
        max_iter = max_iters[env_cfg['env']]
        optim_cfgs = [dict(max_iter=max_iter, algo=algo) for algo in algos]
        # fig = plt.figure()
        compa_optim = compa_algos_env(env_cfg, optim_cfgs, x_axis, y_axis, plot=False, logscale=False, normalize=False)

        sns.lineplot(x=x_axis, y=y_axis, hue='algo', style='algo', data=compa_optim,
                     ax=axs[i],
                     markers=marker_styles, dashes=False, palette=palette, markevery=int(max_iter/15))

        axs[i].set(yscale='log')

        handles, labels = axs[i].get_legend_handles_labels()
        labels = [nice_writing[label] for label in labels]
        axs[i].get_legend().remove()

        x_axis = axs[i].xaxis.get_label().get_text()
        axs[i].set_xlabel(nice_writing[x_axis])

        y_axis = axs[i].yaxis.get_label().get_text()
        axs[i].set_ylabel(nice_writing[y_axis])

        axs[i].set_title(nice_writing[env_cfg['env']])

    fig.legend(handles=handles, labels=labels, loc='center', bbox_to_anchor=(0.5, 0.),
               ncol=len(handles))
    fig.tight_layout()
    fig.show()
    path = '/Users/vincentroulet/git_projects/vincent/ctrl/papers/ilqc_jmlr2022/fig/simple_conv_example.pdf'

    fig.savefig(path, format='pdf', bbox_inches='tight')


def plot_conv_gd_ilqr_ddp_real_car():
    set_plt_params()
    nice_writing = get_nice_writing(alternative=True)
    palette, marker_styles = get_palette_line_styles()

    env_cfg = dict(env='real_car', track='simple', cost='exact', discretization='rk4', reg_bar=0., reg_ctrl=0.,
                   horizon=50, dt=1. / 50)
    approx, step_mode = 'linquad', 'reg'
    max_iter = 200
    algos = ['gd'] + [algo_type + '_' + approx + '_' + step_mode for algo_type in ['classic', 'ddp']]
    optim_cfgs = [dict(max_iter=max_iter, algo=algo) for algo in algos]

    x_axis, y_axis = 'iteration', 'cost'

    compa_optim = compa_algos_env(env_cfg, optim_cfgs, x_axis, y_axis, plot=False, logscale=False, normalize=False)

    compa_optim = compa_optim[compa_optim['iteration'] >= 1]

    fig = plt.figure(figsize=(8, 7))

    ax = sns.lineplot(x=x_axis, y=y_axis, hue='algo', style='algo', data=compa_optim,
                      markers=marker_styles, dashes=False, palette=palette, markevery=int(max_iter / 15))

    ax.set(yscale='log')

    handles, labels = ax.get_legend_handles_labels()
    labels = [nice_writing[label] for label in labels]
    ax.get_legend().remove()

    x_axis = ax.xaxis.get_label().get_text()
    ax.set_xlabel(nice_writing[x_axis])

    y_axis = ax.yaxis.get_label().get_text()
    ax.set_ylabel(nice_writing[y_axis])

    ax.set_title(nice_writing[env_cfg['env']])
    # ax.locator_params(axis="y", numticks=2)
    # ax._request_autoscale_view(tight=True)

    fig.legend(handles=handles, labels=labels, loc='center', bbox_to_anchor=(0.5, 0.),
               ncol=len(handles))
    fig.tight_layout()
    fig.show()
    path = '/Users/vincentroulet/git_projects/vincent/ctrl/papers/ilqc_jmlr2022/fig/hard_conv_example.pdf'
    fig.savefig(path, format='pdf', bbox_inches='tight')


if __name__ == '__main__':
    # plot_conv_all_algos()
    plot_conv_gd_ilqr_iddp()
    # plot_conv_gd_ilqr_ddp_real_car()

