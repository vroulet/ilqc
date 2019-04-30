from copy import deepcopy
from src.pipeline.exp_core import re_start_exp
import numpy as np
from pandas import DataFrame


def grid_search(data_cfg, optim_cfg, stop_exp, params_grid, repeat_exp=1, measure_ref='func_val'):
    """
    Perform a grid search on the parameters given in params_grid that complete None values
     of one of the fields data_cfg or optim_cfg
    :param data_cfg: (dict) contains the default parameters defining the data
    :param optim_cfg: (dict) contains the default parameters defining the optimizer
    :param stop_exp: (dict) or (int) criterium to stop the experiment, e.g. the number of iterations or the time or both
    :param params_grid: (dict) dictionary containing parameters name and their range on which the grid search is done.
                    e.g. params_grid = dict(step_size = [1,2,3], line_search=['armijo', 'wolfe'])
    :param repeat_exp: number of times the experiment must be done
                    (the average best measure is taken to compute best parameters)
    :param measure_ref: (str) measure to assess best parameters in the grid, must be an entry
                              of the info_exp returned in the experiment (see src.pipeline.exp_core.core_exp)
    :return:
        best_params: (dict) best parameters found by the grid search according to measure_ref,
                    e.g. best_params = dict(step_size=3, line_search='wolfe')
        info_exp_grid_search: (list) list of dictionaries, each containing an entry params where the params are recorded
                                    and an entry info_exp where the info on the experiment with this set of parameters
                                    is recorded
    """
    params_list = build_list_from_grid(params_grid)

    default_cfg = dict(data_cfg=data_cfg, optim_cfg=optim_cfg)

    info_exp_grid_search = [
        dict(params=params, info_exp=DataFrame()) for params in params_list]

    best_measure_cfg = list()
    for i, params_to_test in enumerate(params_list):
        cfg_test = set_exp_cfg(default_cfg, params_to_test)

        sum_best_measure = 0
        for k in range(repeat_exp):
            cfg_test['data_cfg']['seed'] = k

            _, info_exp, _ = re_start_exp(stop_exp=stop_exp, **cfg_test)

            info_exp_grid_search[i]['info_exp'] = info_exp_grid_search[i]['info_exp'].append(
                info_exp)

            measure = min(list(info_exp[measure_ref]))
            sum_best_measure += measure

        best_measure_cfg.append(sum_best_measure / repeat_exp)

    idx_best = np.argmin(np.array(best_measure_cfg))
    best_params = params_list[idx_best]

    return best_params, info_exp_grid_search


def build_list_from_grid(params_grid):
    """
    Create a list of parameters from a grid of any size
    :param params_grid: (dict) dictionary containing parameters name and their range on which the grid search is done.
                    e.g. params_grid = dict(step_size = [1,2,3], line_search=['armijo', 'wolfe'])
    :return:
        params_list: (list) list of all possible configurations of the parameters given in the grid,
                    e.g. params_list[0] = dict(step_size=1, line_search='armijo')
    """
    param_sample0 = {key: None for key in params_grid.keys()}
    params_list = [param_sample0]
    for param_name, param_range in params_grid.items():
        new_params_list = []
        for param_sample in params_list:
            for param in param_range:
                new_param_sample = deepcopy(param_sample)
                new_param_sample[param_name] = param
                new_params_list.append(new_param_sample)
        params_list = deepcopy(new_params_list)
    return params_list


def set_exp_cfg(default_exp_cfg, params_to_test):
    """
    Set cfg to test by inserting params in params_to_test into the corresponding entries in the default_cfg
    :param default_exp_cfg: (dict) dictionary with data_cfg, model_cfg, optim_cfg each defining a part of the experiment
                        with corresponding default parameters
    :param params_to_test: (dict) params to test with name and value to insert in the default config
                        and define a config to test
                        e.g. params_to_test = dict(step_size=0, line_search='wolfe')
    :return:
        exp_cfg_to_test: (dict) dictionary with data_cfg, model_cfg, optim_cfg each defining a part of the experiment
                        with params to test incorporated
    """
    for param_name in params_to_test.keys():
        is_in_default_cfg = False
        for cfg_key in default_exp_cfg.keys():
            for default_param_name in default_exp_cfg[cfg_key].keys():
                if default_param_name == param_name:
                    is_in_default_cfg = True
        if not is_in_default_cfg:
            raise ValueError(
                'params to test must be in the default config to be tested')

    exp_cfg_to_test = deepcopy(default_exp_cfg)
    for cfg_key in default_exp_cfg.keys():
        x_cfg = default_exp_cfg[cfg_key]
        for param_key in x_cfg.keys():
            if param_key in params_to_test.keys():
                exp_cfg_to_test[cfg_key][param_key] = params_to_test[param_key]
    return exp_cfg_to_test


def set_cfg(default_cfg, given_params):
    """
    Set cfg with the given_params
    :param default_cfg: (dict) one of data_cfg, model_cfg, optim_cfg dictionaries
    :param given_params: (dict) params to include in the default_cfg
                        e.g. given_params = dict(step_size=0, line_search='wolfe')
    :return:
        cfg_to_test: (dict) updated cfg with the given params
    """
    cfg_to_test = deepcopy(default_cfg)
    for param_name in given_params.keys():
        if param_name not in default_cfg.keys():
            raise ValueError('given params must be in the default config')
    for param_key in default_cfg.keys():
        if param_key in given_params.keys():
            cfg_to_test[param_key] = given_params[param_key]
    return cfg_to_test
