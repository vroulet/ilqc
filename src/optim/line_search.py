import numpy as np
import warnings
import math
import torch
from typing import Callable


def armijo_line_search(func, grad, x_ref, descent_direction, starting_step_size,
                       increasing_factor=2, decreasing_factor=0.5):
    """
    Line-search along descent_direction from x_ref using armijo rule on func whose gradient is given by x_ref
    Use last step-size as starting step-size for the search
    """
    alpha = 0.01
    min_step_size = 1e-12
    stuck = False

    step_size = increasing_factor * starting_step_size

    func_ref = func(x_ref).item()
    gap_target = alpha * torch.sum(grad(x_ref)*descent_direction).item()

    x_try = x_ref + step_size * descent_direction
    while (func(x_try).item() > func_ref + step_size * gap_target or math.isnan(func(x_try).item()))\
            and (step_size > min_step_size):
        step_size = decreasing_factor * step_size
        x_try = x_ref + step_size * descent_direction

    if step_size <= min_step_size:
        stuck = True
        x_try = x_ref

    return x_try, step_size, stuck
