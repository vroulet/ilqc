import numpy as np
import time
import math
import torch
from copy import deepcopy
from pandas import DataFrame

from src.optim.line_search import armijo_line_search


def build_optimizer(pb_oracles, algo='prox_lin', line_search='no', increasing_factor=10, decreasing_factor=0.1,
                    line_search_acc='no', acc_step_size_factor=None, max_iter_LBFGS=10,
                    step_size_init=None):
    """
    Get one of the optimizer defined below
    """
    if algo == 'grad_descent':
        optimizer = GradDescent(pb_oracles, line_search)
    elif algo == 'prox_lin':
        optimizer = ProxLin(pb_oracles, line_search)
    elif algo == 'gauss_newton':
        optimizer = ILQR(pb_oracles, line_search,
                         increasing_factor, decreasing_factor)
    elif algo == 'acc_prox_lin':
        optimizer = AccProxLin(pb_oracles, line_search,
                               line_search_acc, acc_step_size_factor)
    else:
        raise NotImplementedError

    return optimizer


class Optimizer:
    """
    General form of an optimizer used in the experiments
    """

    def __init__(self, pb_oracles):
        #  fixed params
        self.func = pb_oracles['func']
        self.grad = pb_oracles['grad']
        self.prox_lin = pb_oracles['prox_lin']

        # var params
        self.cmd = None
        self.step_size = None

        self.iteration = None
        self.iter_time = None

        self.stopped = None

        # info to plot
        self.info_exp = dict(iteration=list(), iter_time=list(),
                             norm_grad=list(), func_val=list())

    def initialize(self, cmd0, init_step_size):
        """
        Initialize the optimizer
        """
        self.cmd = cmd0
        self.step_size = init_step_size

        self.iteration = 0
        self.iter_time = 0

        self.stopped = None

        self.update_info_record()

    def load_optim_state(self, optim_state):
        """
        Load previously saved optimizer
        """
        self.cmd = optim_state['cmd']
        self.step_size = optim_state['step_size']

        self.iteration = optim_state['iteration']
        self.iter_time = optim_state['iter_time']

        self.stopped = optim_state['stopped']

    def save_optim_state(self):
        """
        Save optimizer state for future computations starting at the iterate saved
        :return:
        """
        optim_state = dict(cmd=self.cmd.data, step_size=self.step_size, iteration=self.iteration,
                           iter_time=self.iter_time, stopped=self.stopped)
        return optim_state

    def step(self):
        """
        One step of the chosen algorithm. Must be override by the chosen instance
        """
        raise NotImplementedError

    def check_stop(self):
        """
        Convergence criterion (here norm of the gradient below some threshold
        :return:
        """
        if torch.norm(self.grad(self.cmd)) < 1e-12:
            print('Algo has converged after {}'.format(self.iteration))
            self.stopped = 'converged'

    def update_info_record(self):
        """
        Record convergence information
        """
        self.info_exp['iteration'].append(deepcopy(self.iteration))
        self.info_exp['iter_time'].append(deepcopy(self.iter_time))

        func_val = self.func(self.cmd).item()
        norm_grad = torch.norm(self.grad(self.cmd)).item()
        self.info_exp['func_val'].append(func_val)
        self.info_exp['norm_grad'].append(norm_grad)

        if self.iteration == 0:
            print('Iter \t Time \t\t Func value \t Norm grad')
        print('{0} \t {1:06.4f} \t {2:06.4f} \t {3:06.4f}'
              .format(self.iteration, self.iter_time, func_val, norm_grad))

    def run_optim(self, max_iter=math.inf, max_time=math.inf):
        """
        Run optimizer until max_iter or max_time
        :return
            sol: (torch.Tensor) optimal command found after a run of the optimizer
            info_exp: (pandas.DataFrame) recorded information on the convergence
            optim_state: (dict) last state of the optimizer
        """
        if max_iter == math.inf and max_time == math.inf:
            raise ValueError
        while self.iter_time < max_time and self.iteration < max_iter:
            t1 = time.time()
            self.step()
            t2 = time.time()
            self.iter_time += t2-t1
            self.iteration += 1

            self.update_info_record()

            self.check_stop()
            if self.stopped is not None:
                break

        sol = deepcopy(self.cmd.data)
        info_exp = deepcopy(DataFrame(self.info_exp))
        optim_state = deepcopy(self.save_optim_state())

        return sol, info_exp, optim_state


class GradDescent(Optimizer):
    """
    Gradient descent on the problem
    """

    def __init__(self, pb_oracles, line_search='armijo', increasing_factor=2, decreasing_factor=0.5):
        super(GradDescent, self).__init__(pb_oracles)
        # fixed params
        self.line_search = line_search
        self.increasing_factor = increasing_factor
        self.decreasing_factor = decreasing_factor

    def step(self):
        stuck = False
        descent_direction = - self.grad(self.x)

        if self.line_search == 'armijo_rule':
            self.cmd, self.step_size, stuck = armijo_line_search(self.func, self.grad, self.cmd, descent_direction,
                                                                 self.step_size,
                                                                 increasing_factor=self.increasing_factor,
                                                                 decreasing_factor=self.decreasing_factor)

        elif self.line_search == 'no':
            self.cmd = self.cmd + self.step_size * descent_direction

        else:
            raise NotImplementedError

        if stuck:
            print('Line-search got stuck')
            self.stopped = 'got stuck'


class ILQR(Optimizer):
    """
    ILQR algorithm (Gauss-Newton method, i.e. prox-linear with an infinite step-size followed by a line-search)
    """

    def __init__(self, pb_oracles, line_search='armijo', increasing_factor=10, decreasing_factor=0.1):
        super(ILQR, self).__init__(pb_oracles)
        # fixed params
        self.line_search = line_search
        self.increasing_factor = increasing_factor
        self.decreasing_factor = decreasing_factor

    def step(self):
        stuck = False

        descent_direction = self.prox_lin(self.cmd, math.inf) - self.cmd

        if self.line_search == 'armijo_rule':
            self.cmd, self.step_size, stuck = armijo_line_search(self.func, self.grad, self.cmd, descent_direction,
                                                                 self.step_size,
                                                                 increasing_factor=self.increasing_factor,
                                                                 decreasing_factor=self.decreasing_factor)

        elif self.line_search == 'no':
            self.cmd = self.cmd + self.step_size * descent_direction

        else:
            raise NotImplementedError

        if stuck:
            print('Line-search got stuck')
            self.stopped = 'got stuck'


class ProxLin(Optimizer):
    """
    Prox-linear algorithm
    """

    def __init__(self, pb_oracles, line_search_method='no'):
        super(ProxLin, self).__init__(pb_oracles)
        # fixed params
        self.line_search_method = line_search_method

    def step(self):
        if self.line_search_method == 'no':
            self.cmd = self.prox_lin(self.cmd, self.step_size)
        else:
            raise NotImplementedError


class AccProxLin(Optimizer):
    """
    Accelerated prox-linear. Makes one standard prox-linear step and an extrapolation step. Take best of both.
    For the implementation and proofs see Roulet, Srinivasa, Drusvyatskiy, Harchaoui,
     Iterative Linearized Control: Stable Algorithms and Complexity Guarantees
    Overloads several functions in Optimizer class to record the extrapolation step
    """

    def __init__(self, pb_oracles, line_search='no', line_search_acc='no', acc_step_size_factor=None):
        super(AccProxLin, self).__init__(pb_oracles)
        # fixed params
        self.line_search = line_search
        self.line_search_acc = line_search_acc

        # var params
        self.cmdz = None
        self.alpha = None
        self.acc_step_size = None
        self.acc_step_size_factor = acc_step_size_factor

    def initialize(self, cmd0, init_step_size):
        super(AccProxLin, self).initialize(cmd0, init_step_size)
        self.cmdz = cmd0
        self.alpha = 2
        self.acc_step_size = init_step_size

    def load_optim_state(self, optim_state):
        super(AccProxLin, self).load_optim_state(optim_state)
        self.cmdz = optim_state['cmdz']
        self.alpha = optim_state['alpha']
        self.acc_step_size = optim_state['acc_step_size']

    def save_optim_state(self):
        optim_state = super(AccProxLin, self).save_optim_state()
        optim_state.update(cmdz=self.cmdz.data, alpha=self.alpha,
                           acc_step_size=self.acc_step_size)
        return optim_state

    def prox_lin_step(self, cmd, step_size, line_search='no'):
        if line_search == 'no':
            y = self.prox_lin(cmd, step_size)
        else:
            raise NotImplementedError
        return y, step_size

    def step(self):
        vk, self.step_size = self.prox_lin_step(
            self.cmd, self.step_size, line_search=self.line_search)

        self.alpha = 2 / (self.iteration + 1)
        yk = self.alpha * self.cmdz + (1 - self.alpha) * self.cmd

        self.acc_step_size = self.step_size*self.acc_step_size_factor
        wk, self.acc_step_size = self.prox_lin_step(
            yk, self.acc_step_size, line_search=self.line_search_acc)

        self.cmdz = self.cmd + (wk - self.cmd) / self.alpha

        if self.func(vk) < self.func(wk):
            self.cmd = vk
        else:
            self.cmd = wk
