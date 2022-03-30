from typing import List
from utils_pipeline.record_exp import set_exp_cfg
from utils_pipeline.grid_search_heatmap import build_grid_from_cfg, build_list_from_grid


def build_list_exp(exp_cfgs: List[dict]) -> list:
    """
    Exp_cfgs is a list of exp_cfg = [dict(data_cfg=..., model_cfg=..., optim_cfg=...), ...]
    Each data_cfg, model_cfg, optim_cfg can have parameters that are list
    For each exp_cfg, we form a list of all combinations of the parameters that are lists
    This function returns a list of all possible configurations exp_cfg
    :param exp_cfgs: [exp_cfg1, exp_cfg2, ...]
                where each exp_cfg is of the form e.g. exp_cfg_i=(data_cfg=..., model_cfg=..., optim_cfg=...)
                (each entry being a dictionary)
                some entries of these dictionaries can be lists that are used to define a grid of parameters
    :returns exp_cfgs_list: list of all possible configurations
    """
    exp_cfgs_list = list()
    if not isinstance(exp_cfgs, list):
        exp_cfgs = [exp_cfgs]
    for exp_cfg in exp_cfgs:
        params_grid = build_grid_from_cfg(exp_cfg)
        params_list = build_list_from_grid(params_grid)
        for params in params_list:
            exp_cfg = set_exp_cfg(exp_cfg, params)
            exp_cfgs_list.append(exp_cfg)
    return exp_cfgs_list


def slice_exp_cfgs_list(exp_cfgs_list: list, nb_slices: int = 999) -> List[list]:
    """
    Given a list, slice the list in nb_slices buckets that are approximately equal
    :param exp_cfgs_list: list
    :param nb_slices: number of slices
    :return: slices: nb_slices slices of the list
    """
    k, m = divmod(len(exp_cfgs_list), nb_slices)
    slices = list((exp_cfgs_list[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(nb_slices)))
    return slices

