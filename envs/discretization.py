from typing import Callable
import torch


def euler(dyn: Callable, state: torch.Tensor, ctrl: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Standard Euler step
    :param dyn: continuous time dynamic, see e.g. the pendulum environment
    :param state: state of the environment
    :param ctrl: control variable applied
    :param dt: time step
    :return: next state of the system
    """
    return state + dt*dyn(state, ctrl)


def runge_kutta4(dyn: Callable, state: torch.Tensor, ctrl: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Runge Kutta discretization with varying control inputs
    :param dyn: continuous time dynamic, see e.g. the pendulum environment
    :param state: state of the environment
    :param ctrl: control variable applied
    :param dt: time step
    :return: next state of the system
    """

    base_dim_ctrl = int(ctrl.shape[0]/3)
    k1 = dyn(state, ctrl[:base_dim_ctrl])
    k2 = dyn(state + dt*k1/2, ctrl[base_dim_ctrl:2*base_dim_ctrl])
    k3 = dyn(state + dt*k2/2, ctrl[base_dim_ctrl:2*base_dim_ctrl])
    k4 = dyn(state + dt*k3, ctrl[2*base_dim_ctrl:])
    return state + dt*(k1 + 2*k2 + 2*k3 + k4)/6


def runge_kutta4_cst_ctrl(dyn: Callable, state: torch.Tensor, ctrl: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Runge Kutta discretization with constant control input
    :param dyn: continuous time dynamic, see e.g. the pendulum environment
    :param state: state of the environment
    :param ctrl: control variable applied
    :param dt: time step
    :return: next state of the system
    """
    k1 = dyn(state, ctrl)
    k2 = dyn(state + dt*k1/2, ctrl)
    k3 = dyn(state + dt*k2/2, ctrl)
    k4 = dyn(state + dt*k3, ctrl)
    return state + dt*(k1 + 2*k2 + 2*k3 + k4)/6
