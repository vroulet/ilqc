"""Discretization methods."""

from typing import Callable
import torch

Dyn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

def euler(dyn: Dyn, state: torch.Tensor, ctrl: torch.Tensor, dt: float) -> torch.Tensor:
    """Standard Euler step.
   
    Args:
      dyn: continuous time dynamic, see e.g. the pendulum environment
      state: state of the environment
      ctrl: control variable applied
      dt: time step

    Returns: next state of the system
    """
    return state + dt*dyn(state, ctrl)


def runge_kutta4(dyn: Dyn, state: torch.Tensor, ctrl: torch.Tensor, dt: float) -> torch.Tensor:
    """Runge Kutta discretization with varying control inputs.

    Args:
      dyn: continuous time dynamic, see e.g. the pendulum environment
      state: state of the environment
      ctrl: control variable applied
      dt: time step

    Returns: next state of the system
    """

    base_dim_ctrl = int(ctrl.shape[0]/3)
    k1 = dyn(state, ctrl[:base_dim_ctrl])
    k2 = dyn(state + dt*k1/2, ctrl[base_dim_ctrl:2*base_dim_ctrl])
    k3 = dyn(state + dt*k2/2, ctrl[base_dim_ctrl:2*base_dim_ctrl])
    k4 = dyn(state + dt*k3, ctrl[2*base_dim_ctrl:])
    return state + dt*(k1 + 2*k2 + 2*k3 + k4)/6


def runge_kutta4_cst_ctrl(dyn: Dyn, state: torch.Tensor, ctrl: torch.Tensor, dt: float) -> torch.Tensor:
    """Runge Kutta discretization with constant control input.
    
    Args:
      dyn: continuous time dynamic, see e.g. the pendulum environment
      state: state of the environment
      ctrl: control variable applied
      dt: time step

    Returns: next state of the system
    """
    k1 = dyn(state, ctrl)
    k2 = dyn(state + dt*k1/2, ctrl)
    k3 = dyn(state + dt*k2/2, ctrl)
    k4 = dyn(state + dt*k3, ctrl)
    return state + dt*(k1 + 2*k2 + 2*k3 + k4)/6
