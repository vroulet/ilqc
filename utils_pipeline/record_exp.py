import os
from copy import deepcopy
import inspect
from functools import wraps
from typing import Any, Callable, List, Tuple


from utils_pipeline.save_load_exp import load_exp, save_exp, var_to_str
from utils_pipeline.set_device_save_load_method import set_save_load_procedure


def run_and_record_exp_wrapper(run_exp: Callable, output_to_input: dict,
                               check_exp_done: Callable[[dict], bool], reload_param: str, results_folder: str) \
                                -> Callable:
    """
    Wrapper that allows to run one experiment, save it and reload it from the last iteration
    :param run_exp: function to run the experiment e.g.
            run_exp(data_cfg, model_cfg, optim_cfg, input1=None, input2=None) -> exp_outputs
            where data_cfg, model_cfg, optim_cfg are dictionaries defining the experiment and input1, input2 are
            optional arguments used to e.g. restart the experiment from the last iteration
    :param output_to_input: dict that maps outputs of the experiments to optional inputs of run_exp
    :param check_exp_done: function to check if the experiment is done given the outputs of an experiment
    :param reload_param: what is the parameter that changes every time the experiment is relaunched,
                         typically the number of iterations (max_iter)
    :param results_folder: path to the folder where to save the results
    :return run_and_record_exp: function to run the experiment with the same nomenclature as run_exp except that now
                                results will be saved and reloaded if the same parameters are given as inputs

    """
    @wraps(run_exp)
    def run_and_record_exp(*cfgs, **exp_cfg):
        exp_cfg = seq_or_keyword_args(run_exp, cfgs, exp_cfg)
        print(*['{0}:{1}'.format(key, value) for key, value in exp_cfg.items()], sep='\n')
        # List are reserved for grid-searches, so if a parameter of the configuration is made of several values,
        # use rather sets
        assert not any([isinstance(value, list) for cfg in exp_cfg.values() for value in cfg.values()])
        exp_outputs = load_exp(exp_cfg, results_folder)
        exp_done = exp_outputs is not None
        if not exp_done:
            exp_outputs, exp_done = re_load_exp(exp_cfg, check_exp_done, reload_param, results_folder)
        if not exp_done:
            if exp_outputs is None:
                exp_outputs = run_exp(**exp_cfg)
            else:
                new_inputs = {val: exp_outputs[key] for key, val in output_to_input.items()}
                exp_outputs = run_exp(**exp_cfg, **new_inputs)
            save_exp(exp_cfg, exp_outputs, results_folder)
        return exp_outputs

    return run_and_record_exp


def run_exp_incrementally_wrapper(run_exp: Callable, output_to_input: dict, check_exp_done: Callable[[dict], bool],
                                  reload_param: str, param_interval_size: int, results_folder: str) -> Callable:
    """
    Wrapper that allows to run one experiment incrementally, i.e., run x iterations, save, then continue
    :param run_exp: function to run the experiment e.g.
            run_exp(data_cfg, model_cfg, optim_cfg, input1=None, input2=None) -> exp_outputs
            where data_cfg, model_cfg, optim_cfg are dictionaries defining the experiment and input1, input2 are
            optional arguments used to e.g. restart the experiment from the last iteration
    :param output_to_input: dict that maps outputs of the experiments to optional inputs of run_exp
    :param check_exp_done: function to check if the experiment is done
    :param reload_param: what is the parameter that changes every time the experiment is relaunched,
                         typically the number of iterations (max_iter)
    :param param_interval_size:
    :param results_folder: path to the folder where to save the results
    :return run_exp_incrementally: function to run the experiment with the same nomenclature as run_exp except that now
                                   intermediate results will be saved

    """
    run_and_record_exp = run_and_record_exp_wrapper(run_exp, output_to_input, check_exp_done,
                                                    reload_param, results_folder)

    @wraps(run_exp)
    def run_exp_incrementally(*cfgs, **exp_cfg):
        exp_cfg = seq_or_keyword_args(run_exp, cfgs, exp_cfg)
        print(*['{0}:{1}'.format(key, value) for key, value in exp_cfg.items()], sep='\n')
        exp_outputs = load_exp(exp_cfg, results_folder)
        exp_done = exp_outputs is not None
        if not exp_done:
            param = 0
            max_param = search_param_exp_cfg(exp_cfg, reload_param)
            while param < max_param:
                temp_exp_cfg = deepcopy(exp_cfg)
                temp_exp_cfg = set_exp_cfg(temp_exp_cfg, {reload_param: min(param + param_interval_size, max_param)})
                exp_outputs = run_and_record_exp(**temp_exp_cfg)
                param = param + param_interval_size
        return exp_outputs
    return run_exp_incrementally


def re_load_exp(exp_cfg: dict, check_exp_done: Callable[[dict], bool], reload_param: str, results_folder: str) \
        -> (dict, bool):
    """
    Given an exp_cfg reload the experiment whose reload_param is the highest
    Typically, find the results found from the last run of a method with the given parameters
    with the highest number of iterations

    Currently only works if reload_param was one parameter of the last dict of exp_cfg, typically, optim_cfg
    :param exp_cfg: dictionary containing the configurations of the given experiment
                    such as exp_cfg=(data_cfg=..., model_cfg=..., optim_cfg=...)
                    (each entry being a dictionary)
    :param check_exp_done: function to check if the experiment is done
    :param reload_param: what is the parameter from which we want to reload the experiment
    :param results_folder: path to the folder where to save the results
    :return
        - exp_outputs - output of the experiment saved in a dictionary
        - exp_done - whether the experiment has already been done or not

    """
    _, load, _ = set_save_load_procedure()
    exp_cfg_folder = results_folder
    for cfg in list(exp_cfg.values())[:-1]:
        exp_cfg_folder += '/{0}'.format(var_to_str(cfg))
    if reload_param not in list(exp_cfg.values())[-1].keys():
        raise ValueError('Can only reload from a parameter defined in the last cfg argument')
    paths = search_paths_similar_cfgs(exp_cfg, exp_cfg_folder, reload_param)

    max_param = search_param_exp_cfg(exp_cfg, reload_param)

    var_param_done = 0
    exp_cfg_to_load = None
    exp_done = False
    exp_outputs = None
    for path in paths:
        with open(path, 'rb') as file:
            exp_cfg_saved = load(file)[0]
            var_param_saved = search_param_exp_cfg(exp_cfg_saved, reload_param)
            if var_param_done < var_param_saved <= max_param:
                var_param_done = var_param_saved
                exp_cfg_to_load = exp_cfg_saved

    if exp_cfg_to_load is not None:
        exp_outputs = load_exp(exp_cfg_to_load, results_folder)
        exp_done = check_exp_done(exp_outputs)
    return exp_outputs, exp_done


def search_paths_similar_cfgs(exp_cfg: dict, root_path: str, variable_param: str) -> List[str]:
    """
    List all paths that are the same as exp_cfg except for the entry 'variable_param'
    Used for example to rerun an experiment when the number of iterations vary
    :param exp_cfg: dictionary containing the configurations of the given experiment
                    such as exp_cfg=(data_cfg=..., model_cfg=..., optim_cfg=...)
                    (each entry being a dictionary)
    :param root_path: path to the folder to search in
    :param variable_param: param that is allowed to be different than the one given in exp_cfg
    :return: paths: list of paths that match exp_cfg except for the parameter variable_param

    """
    _, load, _ = set_save_load_procedure()
    exp_cfg_main = deepcopy(exp_cfg)
    found_param = search_param_exp_cfg(exp_cfg_main, variable_param, del_param=True)
    assert found_param is not None
    all_files = os.listdir(root_path) if os.path.exists(root_path) else list()

    paths = list()
    for file_path in all_files:
        full_path = os.path.join(root_path, file_path)
        with open(full_path, 'rb') as file:
            exp_cfg_saved = load(file)[0]
            search_param_exp_cfg(exp_cfg_saved, variable_param, del_param=True)
            if exp_cfg_main == exp_cfg_saved:
                paths.append(full_path)
    return paths


def search_param_cfg(cfg: dict, param_to_search: str, del_param: bool = False) -> Any:
    """
    Search a parameter in a configuration such as data_cfg (dictionary). Potentially delete that parameter
    :param cfg: dict defining a configuration
    :param param_to_search: parameter to search
    :param del_param: whether to delete the given parameter or not
    :return: found_param: found parameter if present in cfg else None
    """
    found_param = None
    if param_to_search in cfg.keys():
        found_param = deepcopy(cfg[param_to_search])
        if del_param:
            del cfg[param_to_search]
    return found_param


def search_param_exp_cfg(exp_cfg: dict, param_to_search: str, del_param: bool = False) -> Any:
    """
    Given an exp_cfg (i.e. a dict of dicts) find the entry
    in one of the dictionaries that corresponds to 'param_to_search'
    Erase that entry from exp_cfg if del_param=True
    :param exp_cfg: dictionary containing the configurations of the given experiment
                    such as exp_cfg=(data_cfg=..., model_cfg=..., optim_cfg=...)
                    (each entry being a dictionary)
    :poram: param_to_search: parameter to search
    :param del_param: whether to delete the given parameter or not
    :return: found_param: found parameter if present in the exp_cfg else None
    """
    found_param = None
    already_seen = False
    for cfg in exp_cfg.values():
        found_param = search_param_cfg(cfg, param_to_search, del_param)
        assert not (already_seen and found_param is not None)
        already_seen = found_param is not None
    return found_param


def set_cfg(default_cfg: dict, given_params: dict) -> dict:
    """
    Set cfg with the given_params
    :param default_cfg: one of data_cfg, model_cfg, optim_cfg dictionaries
    :param given_params: params to include in the default_cfg
                         e.g. given_params = dict(step_size=0, line_search='wolfe')
    :return: cfg_to_test: updated cfg with the given params
    """
    cfg_to_test = deepcopy(default_cfg)
    for param_key in default_cfg.keys():
        if param_key in given_params.keys():
            cfg_to_test[param_key] = given_params[param_key]
    return cfg_to_test


def set_exp_cfg(default_exp_cfg: dict, given_params: dict) -> dict:
    """
    Set cfg with the given_params
    :param default_exp_cfg: dictionary containing the configurations of the given experiment
                            such as exp_cfg=(data_cfg=..., model_cfg=..., optim_cfg=...)
                            (each entry being a dictionary)
    :param given_params: params to include in the default_cfg
                         e.g. given_params = dict(step_size=0, line_search='wolfe')
    :return: cfg_to_test: updated exp_cfg with the given params
    """
    return {key: set_cfg(cfg, given_params) for key, cfg in default_exp_cfg.items()}


def seq_or_keyword_args(method: Callable, cfgs: Tuple[Any], exp_cfg: dict) -> dict:
    """
    Helper to switch between arguments given as list of dict or dict of dict, returns dict of dict
    :param method: some function
    :param cfgs: list of dict
    :param exp_cfg: dict of dict
    :return: exp_cfg: dict of dict
    """
    if len(cfgs) > 0 and len(exp_cfg) > 0:
        raise ValueError('Current wrappers do not handle both positional and keyword arguments')
    if len(exp_cfg) == 0:
        assert len(cfgs) > 0
        cfg_names = list(inspect.signature(method).parameters.values())
        exp_cfg = {str(cfg_name): cfg for cfg_name, cfg in zip(cfg_names, cfgs)}
    return exp_cfg
