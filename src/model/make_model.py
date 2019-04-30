import math
import torch
from copy import deepcopy

from src.data.utils_data import load_cmd
from src.model.utils_model import conj_grad, chol_problem


def make_model(robot, target_cost_func, data_info):
    """
    Create optimization oracles from the control setting
    :param robot: (torch.nn.Module) Simulated discretized movement of a robot
    :param target_cost_func:  (torch.nn.Module) Final cost on the last state
    :param data_info: (dict) Info on the setting such as the regularization put on a the control varaibles
                            and the dimension of the last state
    :return:
        func: (Callable) Cost of the command provided to the simulated robot to achieve the target (with regularization)
        grad: (Callable) Gradient of the cost of the simulated movement
        prox_lin: (Callable) Prox-linear step on the control setting see below
    """
    dim_state = data_info['dim_state']
    reg_ctrl = data_info['reg_ctrl']

    def func(cmd):
        """
        Cost of the command provided to the simulated robot to achieve the target (with regularization)
        :param cmd (torch.Tensor) matrix of size (dim_ctrl, horizon), each tth column being a control at time t
        return
            total_cost (torch.Tensor) total cost of the simulated movement given by cmd
        """
        robot_ctrls = tensor_to_robot_ctrls(cmd)
        load_cmd(robot, robot_ctrls)

        final_state = robot.forward()
        target_cost = target_cost_func(final_state)
        reg_cost = sum([0.5 * reg_ctrl * torch.sum(ctrl*ctrl)
                        for ctrl in robot.parameters()])
        total_cost = target_cost + reg_cost
        return total_cost

    def grad(cmd):
        """
        Gradient of the cost of the simulated movement
        :param cmd (torch.Tensor) matrix of size (dim_ctrl, horizon), each tth column being a control at time t
                :param cmd (torch.Tensor) matrix of size (dim_ctrl, horizon), each tth column being a control at time t
        return
            grad_cmd (torch.Tensor) gradient of total cost of the simulated movement given by cmd.
            grad_cmd has same dimensions as cmd
        """
        robot_ctrls = tensor_to_robot_ctrls(cmd)
        load_cmd(robot, robot_ctrls)
        robot.zero_grad()

        final_state = robot.forward()
        target_cost = target_cost_func(final_state)
        reg_cost = sum([0.5 * reg_ctrl * torch.sum(ctrl*ctrl)
                        for ctrl in robot.parameters()])
        total_cost = target_cost + reg_cost

        total_cost.backward()
        grad_cmd = [params.grad for params in robot.parameters()]
        grad_cmd = robot_ctrls_to_tensor(grad_cmd)
        return grad_cmd

    def prox_lin(cmd, step_size, max_iter_conj_grad_factor=1):
        """
        Compute a Levenberg-Marquardt step for non-linear_ctrl optimal model pbs
        Formally for functions h,xi,g points x,v, and positive real tau

        prox_lin(x, tau) = x + argmin_y  m_h(xi(x) +  < ∇xi(x), y>) + g(y+x) + 1/(2tau) ||y||^2

        where m_h is a quadratic model on the original last cost.
        The sub-problem is solved by computing its dual solution with a conjugate gradient method
        and plugging it in the primal problem.
        :param cmd: (torch.Tensor) x in above problem
        :param step_size: (float) tau in above problem, can be chosen to be math.inf,
                                which amounts to a gauss-newton step
        :return:
            prox_lin_cmd: (torch.Tensor) prox_lin defined above
        """
        if step_size == math.inf:
            reg_prox_lin = 0
        else:
            reg_prox_lin = 1/step_size

        robot_ctrls = tensor_to_robot_ctrls(cmd)
        load_cmd(robot, robot_ctrls)
        robot.zero_grad()

        final_state = robot.forward()
        # Decomposes the final cost see chol_problem func
        chol_hess, chol_hess_inv_grad = chol_problem(
            final_state, target_cost_func)

        def grad_prox_lin_pg(z):
            """
            Define the gradient of the dual problem by using calls to automatic differentiation oracles
            :param z: (torch.Tensor) dual variable
            :return:
                grad: (torch.Tensor) gradient of the dual problem computed by automatic differentiation
            """
            aux = deepcopy(z.data)
            aux.requires_grad = True
            robot.zero_grad()
            final_state.backward(torch.mv(chol_hess, aux), create_graph=True)

            # out = 0.5 / reg_ctrl * sum([torch.sum(ctrl.grad*ctrl.grad) for ctrl in robot.parameters()])
            out = 0.5/(reg_ctrl + reg_prox_lin)*sum([torch.sum((reg_ctrl * ctrl.data + ctrl.grad)
                                                               * (reg_ctrl * ctrl.data + ctrl.grad))
                                                     for ctrl in robot.parameters()])
            out = out - torch.sum(chol_hess_inv_grad*aux) + \
                0.5 * torch.sum(aux*aux)
            grad = torch.autograd.grad(out, aux)[0]
            return grad

        # Solve dual problem
        dual_sol = conj_grad(chol_hess_inv_grad, grad_prox_lin_pg, max_iter=int(
            max_iter_conj_grad_factor*dim_state))

        # Map dual to primal solution
        robot.zero_grad()
        final_state.backward(torch.mv(chol_hess, dual_sol), create_graph=True)
        prox_lin_cmd = [-1/(reg_ctrl + reg_prox_lin)*(reg_ctrl*ctrl + ctrl.grad)
                        for ctrl in robot.parameters()]
        # prox_lin_cmd = [-1/reg_ctrl*ctrl.grad for ctrl in robot.parameters()]
        prox_lin_cmd = robot_ctrls_to_tensor(prox_lin_cmd)

        # Output proxlin step
        prox_lin_cmd = cmd + prox_lin_cmd
        return prox_lin_cmd

    pb_oracles = dict(func=func, grad=grad, prox_lin=prox_lin)
    return pb_oracles


def tensor_to_robot_ctrls(cmd):
    return [ctrl for ctrl in cmd.t()]


def robot_ctrls_to_tensor(robot_ctrls):
    return torch.stack(robot_ctrls).t()
