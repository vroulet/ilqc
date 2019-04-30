import datetime
import torch
from copy import deepcopy

from src.data.get_data import get_data
from src.model.make_model import make_model
from src.optim.build_optimizer import build_optimizer
from src.utils.save_load import load_exp, save_exp


def core_exp(data_cfg, optim_cfg, stop_exp, last_optim_state=None):
    """
    Wrapper to execute one experiment
    :param data_cfg: (dict) contains the parameters defining the data see get_data function
    :param optim_cfg: (dict) contains the parameters defining the optimizer see build_optimizer fucntion
    :param stop_exp: (dict) or (int) criterium to stop the experiment, e.g. the number of iterations or the time or both
    :param last_optim_state: (dict) parameters of the optimizer to restart the experiment from last iterate
    :return:
        sol: (Any) the solution of the experiment
        info_exp: (DataFrame) a data frame where information on the expeirment has been recorded (such as test accruacy)
        optim_state: (dict) parameters of the optimizer when the experiment has been stopped (such as the last iterate)
    """
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print('\ndata: {0} \noptim: {1}'.format(data_cfg, optim_cfg))
    robot, target_cost_func, data_info = get_data(**data_cfg)

    pb_oracles = make_model(robot, target_cost_func, data_info)

    optimizer = build_optimizer(pb_oracles, **optim_cfg)

    if last_optim_state is not None:
        optimizer.load_optim_state(last_optim_state)
    else:
        cmd0 = torch.zeros(data_info['dim_control'], data_info['horizon'])
        optimizer.initialize(cmd0, optim_cfg['step_size_init'])

    sol, info_exp, optim_state = optimizer.run_optim(**stop_exp)

    return sol, info_exp, optim_state


def re_start_exp(data_cfg, optim_cfg, stop_exp):
    """
    Restart exp from last optim_state found or start exp with the given configuration
    See src.pipeline.exp_core.core_exp for the parameters
    """
    last_sol, last_info_exp, last_optim_state = load_exp(data_cfg, optim_cfg)

    exp_done = False
    if last_optim_state is not None:
        print('Experiment already done until iteration {0}'.format(
            last_optim_state['iteration']))
        if last_optim_state['iteration'] > stop_exp['max_iter']:
            exp_done = True
        if 'stopped' in last_optim_state.keys() and last_optim_state['stopped'] is not None:
            exp_done = True
            print('Optimizer has {0}'.format(last_optim_state['stopped']))

    if not exp_done:
        sol, info_exp, optim_state = core_exp(data_cfg, optim_cfg, stop_exp,
                                              last_optim_state=last_optim_state)
        if last_info_exp is not None:
            info_exp = last_info_exp.append(info_exp)

        save_exp(data_cfg, optim_cfg, sol, info_exp, optim_state)

    else:
        sol = last_sol
        info_exp = last_info_exp
        optim_state = last_optim_state

    return sol, info_exp, optim_state


def safe_exp(data_cfg, optim_cfg, stop_exp, iteration_interval=100):
    """
    Make regular saves of the experiment along the computations. Provide robustness to unexpected stops.
    See src.pipeline.exp_core.core_exp for the parameters
    """
    iteration = 0
    info_exp = None
    optim_state = None
    sol = None
    while iteration < stop_exp['max_iter']:
        temp_stop_exp = deepcopy(stop_exp)
        temp_stop_exp['max_iter'] = min(
            iteration + iteration_interval, stop_exp['max_iter'])
        sol, info_exp, optim_state = re_start_exp(
            data_cfg, optim_cfg, temp_stop_exp)
        iteration = optim_state['iteration']
    return sol, info_exp, optim_state
