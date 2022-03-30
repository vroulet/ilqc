import torch


def roll_out_lin(traj, gains, stepsize=1.):
    dim_state, horizon = traj[0].shape, len(traj)-1
    dim_var = gains[0]['offset_ctrl'].shape[0]

    diff_x = torch.zeros(dim_state)
    diff_vars = torch.zeros(horizon, dim_var)

    for t in range(horizon):
        A = traj[t + 1].grad_dyn_state.t()
        B = traj[t + 1].grad_dyn_ctrl.t()

        diff_var = gains[t]['gain_ctrl'].mv(diff_x) + stepsize*gains[t]['offset_ctrl']
        diff_vars[t] = diff_var
        diff_x = A.mv(diff_x) + B.mv(diff_var)
    return diff_vars


def roll_out_exact(env, traj, gains, cmd, stepsize=1.):
    dim_state, horizon = traj[0].shape, len(traj) - 1
    diff_x, state = torch.zeros(dim_state), env.init_state
    diff_vars = torch.zeros_like(cmd)
    diff_vars.requires_grad = state.requires_grad = False

    for t in range(horizon):
        diff_var = gains[t]['gain_ctrl'].mv(diff_x) + stepsize * gains[t]['offset_ctrl']
        diff_vars[t] = diff_var
        state = env.discrete_dyn(state, cmd[t].data + diff_var, t)
        diff_x = state - traj[t+1].data
    return diff_vars



