import inspect
import os
from typing import Any

import torch

from utils_pipeline.set_device_save_load_method import set_save_load_procedure


def save_exp(
    exp_cfg: dict[str, Any], exp_outputs: dict[str, Any], results_folder: str
) -> None:
    """Save the results of the experiment exp_cfg with the nomenclature explained in get_path_exp.

    Args:
      exp_cfg: dictionary containing the configurations of the given experiment
                    such as exp_cfg=(data_cfg=..., model_cfg=..., optim_cfg=...)
                    (each entry being a dictionary)
      exp_outputs: dictionary containing the outputs of the experiments
      results_folder: folder in which to save the results
    """
    save, _, format_files = set_save_load_procedure()
    path = results_folder
    for cfg in exp_cfg.values():
        path += "/{0}".format(var_to_str(cfg))
    path = path + format_files
    assert not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        save([exp_cfg, exp_outputs], file)


def load_exp(exp_cfg: dict[str, Any], results_folder: str) -> dict:
    """Load the result of a method applied to exp_cfg if teh result had already been done.

    Args:
      exp_cfg: dictionary containing the configurations of the given experiment
                 such as exp_cfg=(data_cfg=..., model_cfg=..., optim_cfg=...)
                 (each entry being a dictionary)
      results_folder: folder in which to save the results

    Returns:
      exp_outputs: dictionary containing the outputs of the experiments
    """
    _, load, format_files = set_save_load_procedure()
    path = results_folder
    for cfg in exp_cfg.values():
        path += "/{0}".format(var_to_str(cfg))
    path = path + format_files
    exp_outputs = None
    if os.path.exists(path):
        with open(path, "rb") as file:
            loaded = load(file)
            exp_outputs = loaded[1]
    return exp_outputs


def var_to_str(var: Any) -> str:
    """Given an object var generate a str that identifies this object and can easily be readable.

    Args:
      var:  object from which we want to generate an automatic nomenclature

    Returns:
      var_str: string that corresponds to the given object
    """
    translate_table = {ord(c): None for c in ",()[]"}
    translate_table.update({ord(" "): "_"})

    if type(var) == dict:
        sortedkeys = sorted(var.keys(), key=lambda x: x.lower())
        var_str = [
            key + "_" + var_to_str(var[key])
            for key in sortedkeys
            if var[key] is not None
        ]
        var_str = "_".join(var_str)
    elif inspect.isclass(var):
        raise NotImplementedError("Do not give classes as items in cfg inputs")
    elif type(var) in [list, set, frozenset, tuple]:
        value_list_str = [var_to_str(item) for item in var]
        var_str = "_".join(value_list_str)
    elif isinstance(var, float):
        var_str = "{0:1.2e}".format(var)
    elif isinstance(var, int):
        var_str = str(var)
    elif isinstance(var, str):
        var_str = var
    elif var is None:
        var_str = str(var)
    elif isinstance(var, torch.Tensor):
        # todo: use norm of the tensor as its signature for saving it, avoid if possible
        var_str = "{0:.6e}".format(torch.norm(var).item())
    else:
        print(type(var))
        raise NotImplementedError
    return var_str


def save_entry(
    exp_cfg: dict[str, Any], results: dict[str, Any], path: str
) -> None:
    """Save the best_params of a grid search done on exp_cfg.

    Args:
      exp_cfg: dictionary containing the configurations of the given experiment
                    such as exp_cfg=(data_cfg=..., model_cfg=..., optim_cfg=...)
                    (each entry being a dictionary)
      results: best parameters from the varying parameters in the exp_cfg
                    (defined by the entries of the dictionary that are lists)
                    see grid_search for more details
      path: path to a file where results are saved
    """
    save, load, _ = set_save_load_procedure()
    new_entry = dict(exp_cfg=exp_cfg, results=results)
    if not os.path.exists(path):
        with open(path, "wb") as file:
            save([new_entry], file)
    else:
        with open(path, "rb") as file:
            entries = load(file)
        for entry in entries:
            assert entry["exp_cfg"] != new_entry["exp_cfg"]
        entries.append(new_entry)
        with open(path, "wb") as file:
            save(entries, file)


def load_entry(exp_cfg: dict[str, Any], path: str) -> dict:
    """Load an experiment.

    Look at a table of all grid-searches run before and check if the one given by exp_cfg exp_cfg has already been done
    If yes return that search.

    Args:
      exp_cfg: dictionary containing the configurations of the given experiment
                    such as exp_cfg=(data_cfg=..., model_cfg=..., optim_cfg=...)
                    (each entry being a dictionary)
      path: path to a file where results are saved

    Returns:
      results: best parameters from the varying parameters in the exp_cfg
                     (defined by the entries of the dictionary that are lists)
                     see grid_search for more details
    """
    _, load, _ = set_save_load_procedure()
    results = None
    if os.path.exists(path):
        with open(path, "rb") as file:
            entries = load(file)
        for entry in entries:
            if entry["exp_cfg"] == exp_cfg:
                assert results is None
                results = entry["results"]
    return results


def erase_entry(exp_cfg: dict[str, Any], path: str) -> None:
    """Erase a specific entry of the file containing the grid-searches.

    Args:
      exp_cfg: dictionary containing the configurations of the given experiment
                    such as exp_cfg=(data_cfg=..., model_cfg=..., optim_cfg=...)
                    (each entry being a dictionary)
      path: path to a file where results are saved
    """
    save, load, _ = set_save_load_procedure()
    with open(path, "rb") as file:
        entries = load(file)
    for i, entry in enumerate(entries):
        if entry["exp_cfg"] == exp_cfg:
            entries.pop(i)
    with open(path, "wb") as file:
        save(entries, file)
