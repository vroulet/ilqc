from exp_cfg import plots_folder
from src.utils.plot import plot_exp
from src.pipeline.grid_search import grid_search, set_cfg
from src.pipeline.exp_core import re_start_exp
import matplotlib.pyplot as plt
from pandas import DataFrame
from copy import deepcopy
import math
import argparse

# Select the model you want to test: inverse pendulum or the two-links arm
# see src.data.get_data for detailed configuration

parser = argparse.ArgumentParser(description='Regularized ILQR comparisons')
parser.add_argument('--ctrl_setting', default='inverse_pendulum', type=str,
                    help='Artificial control setting')
parser.add_argument('--horizon', default=100, type=int,
                    help='Horizon of the discrete control problem')
parser.add_argument('--target_goal', default='swing_up', type=str,
                    help='Cost on the final state of the trajectory')
args = parser.parse_args()

data_cfg = dict(ctrl_setting=args.ctrl_setting, horizon=args.horizon, seed=0,
                target_goal=args.target_goal, reg_ctrl=0.01)

# Grid search on algorithms for the step-sizes
# see src.optim.build_optimizer for more information
optim_cfgs = list()
stop_exp_gs = dict(max_iter=5)

optim_cfg = dict(algo='prox_lin', line_search='no', step_size_init=None)
params_grid = dict(step_size_init=[2**i for i in range(12)])
best_params_pl, _ = grid_search(data_cfg, optim_cfg, stop_exp_gs, params_grid)
optim_cfg = set_cfg(optim_cfg, best_params_pl)
optim_cfgs.append(optim_cfg)

optim_cfg = dict(algo='gauss_newton', line_search='armijo_rule', step_size_init=None,
                 increasing_factor=None, decreasing_factor=None)
params_grid = dict(step_size_init=[2**i for i in range(12)],
                   increasing_factor=[2],
                   decreasing_factor=[0.5])
best_params, _ = grid_search(data_cfg, optim_cfg, stop_exp_gs, params_grid)
optim_cfg = set_cfg(optim_cfg, best_params)
optim_cfgs.append(optim_cfg)

optim_cfg = dict(algo='acc_prox_lin', line_search='no', line_search_acc='no',
                 step_size_init=best_params_pl['step_size_init'], acc_step_size_factor=None)
params_grid = dict(acc_step_size_factor=[2**i for i in range(-5, 5)])
best_params, _ = grid_search(data_cfg, optim_cfg, stop_exp_gs, params_grid)
optim_cfg = set_cfg(optim_cfg, best_params)
optim_cfgs.append(optim_cfg)

stop_exp = dict(max_iter=20)

# Run the experiments
total_info_exp = DataFrame()
for optim_cfg in optim_cfgs:
    _, info_exp, _ = re_start_exp(data_cfg, optim_cfg, stop_exp)
    info_exp['algo'] = [optim_cfg['algo']]*len(info_exp)
    total_info_exp = total_info_exp.append(
        deepcopy(info_exp), ignore_index=True)


# Run a bit longer to find minimum value of the function
stop_exp_ref = dict(max_iter=int(stop_exp['max_iter']*1.5))

min_func = math.inf
for optim_cfg in optim_cfgs:
    _, info_exp, _ = re_start_exp(data_cfg, optim_cfg, stop_exp_ref)
    min_func = min(list(info_exp['func_val']) + [min_func])

# Normalize the plots and plot
normalized_info_exp = deepcopy(total_info_exp)
normalized_info_exp['func_val'] = normalized_info_exp['func_val'] - \
    min_func + 1e-12

x_axis_plots = ['iteration', 'iter_time']
y_axis_plots = ['func_val', 'norm_grad']

fig, axes = plt.subplots(len(x_axis_plots), len(
    y_axis_plots), figsize=(16, 12))
for i, x_axis in enumerate(x_axis_plots):
    for j, y_axis in enumerate(y_axis_plots):
        plot_exp(normalized_info_exp, x_axis, y_axis, hue='algo',
                 log_scale=True, normalize=True, plot_min_until_now=True, ax=axes[i, j])

fig.savefig('{0}/{1}_{2}_horizon_{3}.pdf'.format(plots_folder, data_cfg['ctrl_setting'],
                                                 data_cfg['target_goal'], data_cfg['horizon']), format='pdf')
plt.show()
