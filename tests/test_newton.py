import sys
import time

sys.path.append('.')

import torch

from algorithms.algos_steps import newton_oracle, classic_oracle
from envs.pendulum import Pendulum

torch.set_default_dtype(torch.double)
torch.manual_seed(1)

def test_newton():
  env = Pendulum(horizon=100, stay_put_time=1.)
  cmd = 0.1*torch.randn(env.horizon, env.dim_ctrl, requires_grad=True)
  traj, costs = env.forward(cmd, approx='quad')
  stepsize = 1.
  ilqc_oracle = classic_oracle(traj, costs, approx='quad', step_mode='dir', handle_bad_dir='check_total_value')
  auto_diff_oracle = newton_oracle(env, cmd)
  cmd_ilqc, _ = ilqc_oracle(stepsize)
  cmd_newton, _ = auto_diff_oracle(stepsize)
  print(f'Difference ilqc autodiff: {torch.linalg.norm(cmd_ilqc-cmd_newton)}')

if __name__ == '__main__':
  test_newton()