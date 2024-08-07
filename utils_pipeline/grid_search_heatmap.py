"""Utilities for grid searches and heatmaps."""

from copy import deepcopy
from typing import Any, Callable

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from pandas import DataFrame
import seaborn as sns

from utils_pipeline.save_load_exp import save_entry, load_entry
from utils_pipeline.record_exp import set_exp_cfg, seq_or_keyword_args

CRED, CEND = "\033[91m", "\033[0m"


def grid_search_wrapper(
    run_exp: Callable[..., Any],
    measure_perf: Callable[[dict[str, Any]], float],
    direction: str,
    grid_search_results_path: str,
) -> Callable[..., Any]:
    """Wrapper that defines a grid-search function for the given template experiment given in run_exp.
    
    Args:
      run_exp: function to run the experiment e.g.
          run_exp(data_cfg, model_cfg, optim_cfg, input1=None, input2=None) -> exp_outputs
          where data_cfg, model_cfg, optim_cfg are dictionaries defining the experiment and input1, input2 are
          optional arguments used to e.g. restart the experiment from the last iteration
          Here better to have the run_and_record_exp version of run_exp as explained in record_exp
      measure_perf: function that measure the performance of the given experiment given its outputs
      direction: 'min' or 'max' i.e. take the minimum or the maximum of the measured performances
      grid_search_results_path: path to save the best computed params

    Returns:
      grid_search: function that can run a grid search on all parameters of a given exp_cfg that are expressed
        as lists, see below
    """

    def grid_search(*cfgs, **exp_cfg):
        """Grid search.

        Take e.g. exp_cfg=dict(data_cfg=..., model_cfg=..., optim_cfg=...)
        where optim_cfg = dict(algo='sgd', lr= [1, 2, 3]
        Create a list of experiments from this setting such as
        exp_cfg1=dict(data_cfg=..., model_cfg=..., optim_cfg=dict(algo='sgd', lr=1)
        exp_cfg2=dict(data_cfg=..., model_cfg=..., optim_cfg=dict(algo='sgd', lr=2)
        ...
        Run all the exp_cfgs, measure their performance, take the best, record the search.

        The inputs can either be a tuple as
        grid_search(data_cfg, model_cfg, optim_cfg)
        or by keyword arguments grid_search(data_cfg=..., model_cfg=..., optim_cfg=....)
        """
        exp_cfg = seq_or_keyword_args(run_exp, cfgs, exp_cfg)

        best_params = load_entry(exp_cfg, grid_search_results_path)
        if best_params is None:
            params_grid = build_grid_from_cfg(exp_cfg)
            assert len(params_grid) > 0

            params_list = build_list_from_grid(params_grid)
            best_measure_cfg = list()
            for i, params in enumerate(params_list):
                search_exp_cfg = set_exp_cfg(exp_cfg, params)
                print(
                    CRED
                    + "Grid search percentage {0:2.0f}".format(
                        float(i / len(params_list)) * 100
                    )
                    + CEND
                )
                exp_outputs = run_exp(**search_exp_cfg)
                measure = measure_perf(exp_outputs)
                best_measure_cfg.append(measure)

            if direction == "min":
                idx_best = int(np.argmin(np.array(best_measure_cfg)))
            elif direction == "max":
                idx_best = int(np.argmax(np.array(best_measure_cfg)))
            else:
                raise NotImplementedError
            best_params = params_list[idx_best]

            save_entry(exp_cfg, best_params, grid_search_results_path)

        print(CRED + "best params:" + str(best_params) + CEND)
        return best_params

    return grid_search


def compute_heatmap_wrapper(
    run_exp: Callable[..., Any],
    measure_perf: Callable[[dict], str],
    heatmap_results_path: str,
) -> Callable[..., Any]:
    """Wrapper that defines a heatmap function for the given template experiment given in run_exp.
   
    Args:
      run_exp: function to run the experiment e.g.
        run_exp(data_cfg, model_cfg, optim_cfg, input1=None, input2=None) -> exp_outputs
        where data_cfg, model_cfg, optim_cfg are dictionaries defining the experiment and input1, input2 are
        optional arguments used to e.g. restart the experiment from the last iteration
        Here better to have the run_and_record_exp version of run_exp as explained in record_exp
      measure_perf: function that measure the performance of the given experiment given its outputs
      heatmap_results_path: path to save the results
    
    Returns:
      compute_heatmap: function that can run a heatmap for the given template experiment
    """

    def compute_heatmap(
        x_param_name: str, y_param_name: str, *cfgs, **exp_cfg
    ):
        """Compute heatmap.

        Compute a heatmap of the performance of the experiment along the x_param and y_param axes
        exp_cfg is a dict of dict where in one of those dicts the key is the chosen x_param with its value being a
        list of corresponding parameters to review. Same for y_param
        Outputs a dict of list ready to be transformed in a DataFrame and plotted
        """
        exp_cfg = seq_or_keyword_args(run_exp, cfgs, exp_cfg)

        heatmap = load_entry(exp_cfg, heatmap_results_path)
        if heatmap is None:
            params_grid = build_grid_from_cfg(exp_cfg)
            for key in params_grid.keys():
                if key not in [x_param_name, y_param_name]:
                    raise ValueError

            x_params = params_grid[x_param_name]
            y_params = params_grid[y_param_name]

            heatmap = {
                x_param_name: list(),
                y_param_name: list(),
                "measure": list(),
            }
            counter = 0
            for x_param in x_params:
                for y_param in y_params:
                    params = {x_param_name: x_param, y_param_name: y_param}
                    search_exp_cfg = set_exp_cfg(exp_cfg, params)
                    print(
                        *[
                            "{0}:{1}".format(key, value)
                            for key, value in search_exp_cfg.items()
                        ],
                        sep="\n",
                    )

                    exp_outputs = run_exp(**search_exp_cfg)
                    measure = measure_perf(exp_outputs)
                    heatmap[x_param_name].append(x_param)
                    heatmap[y_param_name].append(y_param)
                    heatmap["measure"].append(measure)
                    print(
                        CRED + f"Heatmap percentage "
                        f"{float(counter/len(build_list_from_grid(params_grid)))*100:2.0f}"
                        + CEND
                    )
                    counter += 1
            save_entry(exp_cfg, heatmap, heatmap_results_path)
        return heatmap

    return compute_heatmap


def plot_heatmap(heatmap: dict[str, Any]) -> tuple[Figure, Axes]:
    """Plot the heatmap.
    
    Args:
      heatmap: dict of lists containing the performances

    Returns:
        - fig: figure
        - ax: axis
    """
    x_param_name, y_param_name = list(heatmap.keys())[:2]
    heatmap = DataFrame(heatmap)
    heatmap = heatmap.pivot(x_param_name, y_param_name, "measure")
    ax = sns.heatmap(heatmap)
    fig = ax.get_figure()
    fig.canvas.draw()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    return fig, ax


def build_list_from_grid(params_grid: dict[str, Any]) -> list[dict[str, Any]]:
    """Create a list of parameters from a grid of any size.

    Args:
      params_grid: dictionary containing parameters name and their range on which the grid search is done.
        e.g. params_grid = dict(step_size = [1,2,3], line_search=['armijo', 'wolfe'])
    
    Returns:
      params_list: list of all possible configurations of the parameters given in the grid,
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


def build_grid_from_cfg(exp_cfg: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Scan the exp_cfg and extract entries that are lists to be searched on by e.g. a grid-search.

    Args:
      exp_cfg: such as exp_cfg=dict(data_cfg=..., model_cfg=..., optim_cfg=...)
        (each entry being a dictionary)
        some entries of these dictionaries must be lists that are used to define a grid of parameters

    Returns:
      params_grid: dictionary of the form dict(param1=[], param2=[], ...)
        where each param_i corresponds to one parameter in the exp_cfg that was given in the form of a list
        the list is then the corresponding value of param_i in this dictionary
    """
    params_grid = dict()
    for cfg in exp_cfg.values():
        for key, value in cfg.items():
            if isinstance(value, list):
                params_grid.update({key: value})
    return params_grid
