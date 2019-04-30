import inspect
import os
import pathlib
from exp_cfg import info_exp_folder
import torch


def load_exp(data_cfg, optim_cfg):
    """
    Load experiment corresponding to the given configurations
    See src.pipeline.exp_core.core_exp for the parameters
    """
    file_path = make_path(data_cfg, optim_cfg)

    if os.path.isfile(file_path):
        with open(file_path, 'rb') as file:
            [_,  _, sol, info_exp, optim_state] = torch.load(file)
    else:
        info_exp = None
        optim_state = None
        sol = None

    return sol, info_exp, optim_state


def save_exp(data_cfg, optim_cfg, sol, info_exp, optim_state):
    """
    Save experiment in a systematic way
    See src.pipeline.exp_core.core_exp for the parameters
    """
    print('Saving info_exp')

    file_path = make_path(data_cfg, optim_cfg, create_dir=True)
    with open(file_path, 'wb') as file:
        torch.save([data_cfg, optim_cfg, sol, info_exp, optim_state], file)


def make_path(data_cfg, optim_cfg, create_dir=False):
    """
    Get path to config, use info_exp_folder from exp.exp_cfg to store the results
    :param data_cfg, optim_cfg: (dict) see src.pipeline.exp_core.core_exp
    :param create_dir: (bool) True if one wants to create the directory path (for saving)
    :return:
        exp_path: (str) systematic path to save load experiment
    """
    data_cfg_dir = var_to_str(data_cfg)
    # model_cfg_dir = var_to_str(model_cfg)
    # dir_path = '{0}/{1}/{2}'.format(info_exp_folder, data_cfg_dir, model_cfg_dir)
    dir_path = '{0}/{1}'.format(info_exp_folder, data_cfg_dir)

    if create_dir:
        try:
            pathlib.Path(dir_path).mkdir(parents=True)
        except OSError:
            pass

    file_name = var_to_str(optim_cfg)

    exp_path = '{0}/{1}.pickle'.format(dir_path, file_name)
    return exp_path


def var_to_str(var):
    """
    Returns a string from a variable (such as a dict of parameters)
    :param var: (Any) parameter to transform in a string
    :return:
        var_str: (str) String defining the paramter
    """
    translate_table = {ord(c): None for c in ',()[]'}
    translate_table.update({ord(' '): '_'})

    if type(var) == dict:
        sortedkeys = sorted(var.keys(), key=lambda x: x.lower())
        var_str = [key + '_' + var_to_str(var[key])
                   for key in sortedkeys if var[key] is not None]
        var_str = '_'.join(var_str)
    elif inspect.isclass(var):
        raise NotImplementedError('Do not give as inputs in cfg inputs')
    elif type(var) in [list, set, frozenset]:
        value_list_str = [var_to_str(item) for item in var]
        var_str = '_'.join(value_list_str)
    elif isinstance(var, float):
        var_str = '{0:1.2e}'.format(var)
    elif isinstance(var, int):
        var_str = str(var)
    elif isinstance(var, str):
        var_str = var
    else:
        print(type(var))
        raise NotImplementedError

    return var_str
