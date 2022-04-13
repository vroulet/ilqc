import math
import numpy as np
import torch
from envs.forward import DiffEnv
from envs.discretization import euler, runge_kutta4, runge_kutta4_cst_ctrl
from envs.torch_utils import smooth_relu
from envs import rendering


class Pendulum(DiffEnv):
    def __init__(self, reg_speed=0.1, reg_ctrl=1e-6, dt=0.05, horizon=40, stay_put_time=None, seed=0):
        super(Pendulum, self).__init__()
        # env parameters
        self.horizon, self.dt = horizon, dt
        self.dim_ctrl, self.dim_state = 1, 2
        self.init_state, self.init_time_iter = torch.tensor(([math.pi, 0.])), 0
        if seed != 0:
            torch.manual_seed(seed)
            self.init_state[0] = self.init_state[0] + 1e0*torch.randn(1)

        # cost parameters
        self.reg_speed, self.reg_ctrl = reg_speed, reg_ctrl
        self.stay_put_time_start_iter = horizon-int(stay_put_time/dt) if stay_put_time is not None else horizon

        # physics parameters
        self.g, self.m, self.l, self.mu = 10, 1., 1., 0.01

        # rendering
        self.pole_transform = None

    def dyn(self, state, ctrl):
        th, thdot = state
        g, m, l, mu = self.g, self.m, self.l, self.mu
        dthdot = -g/l * torch.sin(th + math.pi) - mu/(m*l**2) * thdot + 1/(m*l**2) * ctrl
        dth = thdot.unsqueeze(-1)
        dt_state = torch.stack((dth, dthdot)).view(-1)
        return dt_state

    def discrete_dyn(self, state, ctrl, time_iter):
        return euler(self.dyn, state, ctrl, self.dt)

    def costs(self, next_state, ctrl, time_iter):
        cost_next_state = self.cost_state(next_state, time_iter)
        return cost_next_state, self.cost_ctrl(ctrl)

    def cost_ctrl(self, ctrl):
        return self.reg_ctrl * ctrl ** 2

    def cost_state(self, state, time_iter):
        if time_iter >= self.stay_put_time_start_iter:
            cost_state = (state[0]) ** 2 + self.reg_speed * state[1] ** 2
        else:
            cost_state = torch.tensor(0.)
        return cost_state

    def reset(self, requires_grad=False):
        self.time_iter = self.init_time_iter
        self.state = self.init_state
        self.state.requires_grad = requires_grad
        return self.state

    def set_viewer(self):
        l = 2 * self.l
        self.viewer = rendering.Viewer(500, 500)
        self.viewer.set_bounds(-1.5 * l, 1.5 * l, -1.5 * l, 1.5 * l)
        rod = rendering.make_capsule(l, 0.1 * l)
        rod.set_color(0., 0., 0.)
        self.pole_transform = rendering.Transform()
        rod.add_attr(self.pole_transform)
        self.viewer.add_geom(rod)

    def render(self, title=None):
        if self.viewer is None:
            self.set_viewer()
        np_state = self.state.numpy()
        self.pole_transform.set_rotation(np_state[0] + np.pi/2)
        return self.viewer.render(title=title)


class CartPendulum(DiffEnv):
    def __init__(self, reg_speed=0.1, reg_ctrl=1e-6, reg_barrier=1., dt=0.05, horizon=40, stay_put_time=None,
                 discretization='euler', x_limits=None, seed=0):
        super(CartPendulum, self).__init__()

        # env parameters
        self.horizon, self.dt, self.discretization = horizon, dt, discretization
        self.dim_ctrl, self.dim_state = 1, 4
        if discretization == 'rk4':
            self.dim_ctrl *= 3
        self.init_state, self.init_time_iter = torch.tensor(([0., 0., 0., 0.])), 0
        if seed != 0:
            torch.manual_seed(seed)
            self.init_state[0] = self.init_state[0] + 1e0*torch.randn(1)

        # cost parameters
        self.reg_speed, self.reg_ctrl, self.reg_barrier = reg_speed, reg_ctrl, reg_barrier
        self.stay_put_time_start_iter = horizon - int(stay_put_time / dt) if stay_put_time is not None else horizon

        # physics parameters
        self.g, self.M, self.m, self.b, self.I, self.l = 10, 0.5, 0.2, 0.1, 0.006, 0.3

        # rendering parameters
        self.x_limits = x_limits
        self.pole_transform = self.cart_transform = None

    def dyn(self, state, ctrl):
        x, xdot, th, thdot = state
        g, M, m, b, I, l = self.g, self.M, self.m, self.b, self.I, self.l
        # dthdot = (-g / l * torch.sin(th + math.pi) - mu/(m*l**2) + 1/ (m * l ** 2) * ctrl)
        # The system can be written as
        # Az = b
        # for z = (dxdot, dthdot) and A, b given by physics laws (A is symmetric)
        # The following code inverts A to get the expression of z
        a11, a22 = M+m, I+m*l**2
        a12 = m*l*torch.cos(th)
        detA = I*(M+m) + m*l**2*M + m**2*l**2*torch.sin(th)**2
        b1 = m*l*thdot**2*torch.sin(th) - b*xdot + ctrl
        b2 = -m*g*l*torch.sin(th)
        dxdot = (a22*b1 - a12*b2)/detA
        dthdot = (-a12*b1 + a11*b2)/detA
        dx = xdot.unsqueeze(-1)
        dth = thdot.unsqueeze(-1)
        dt_state = torch.stack((dx, dth, dxdot, dthdot)).view(-1)
        return dt_state

    def discrete_dyn(self, state, ctrl, time_iter):
        if self.discretization == 'euler':
            next_state = euler(self.dyn, state, ctrl, self.dt)
        elif self.discretization == 'rk4':
            next_state = runge_kutta4(self.dyn, state, ctrl, self.dt)
        elif self.discretization == 'rk4cst':
            next_state = runge_kutta4_cst_ctrl(self.dyn, state, ctrl, self.dt)
        else:
            raise NotImplementedError
        return next_state

    def costs(self, next_state, ctrl, time_iter):
        return self.cost_state(next_state, time_iter), self.cost_ctrl(ctrl)

    def cost_ctrl(self, ctrl):
        if ctrl.shape[0] > 0:
            cost_ctrl = ctrl.dot(ctrl)
        else:
            cost_ctrl = ctrl**2
        return self.reg_ctrl * cost_ctrl

    def cost_state(self, state, time_iter):
        if time_iter >= self.stay_put_time_start_iter:
            cost_state = (state[1]+math.pi) ** 2 + self.reg_speed * state[3] ** 2
        else:
            cost_state = torch.tensor(0.)
        if self.x_limits is not None:
            # cost_barrier = 1e-6*(torch.log(state[0] - self.x_limits[0]) + torch.log(self.x_limits[1] - state[0]))
            cost_barrier = self.reg_barrier*(smooth_relu(-(state[0] - self.x_limits[0]))
                                             + smooth_relu(-(self.x_limits[1] - state[0])))
            cost_state = cost_state + cost_barrier
        return cost_state

    def set_viewer(self):
        self.viewer = rendering.Viewer(500, 500)
        l = 2 * self.l
        if self.x_limits is not None:
            xmin, xmax = self.x_limits
        else:
            xmin, xmax = -4 * l, 4 * l
        self.viewer.set_bounds(xmin - l, xmax + l, xmin - l, xmax + l)

        cart_size = 2 * l / 5
        lc, rc, tc, bc = -cart_size, cart_size, 0., -cart_size
        cart = rendering.make_polygon([(lc, bc), (lc, tc), (rc, tc), (rc, bc)])
        cart.set_color(0.588, 0.294, 0.)
        self.cart_transform = rendering.Transform()
        cart.add_attr(self.cart_transform)
        self.viewer.add_geom(cart)

        rod = rendering.make_capsule(l, .2 * l)
        rod.set_color(0., 0., 0.)
        self.pole_transform = rendering.Transform()
        rod.add_attr(self.pole_transform)

        self.viewer.add_geom(rod)

    def render(self, title=None):
        if self.viewer is None:
            self.set_viewer()
        np_state = self.state.numpy()
        self.cart_transform.set_translation(np_state[0], 0)
        self.pole_transform.set_translation(np_state[0], 0)
        self.pole_transform.set_rotation(np_state[1] - np.pi/2)

        return self.viewer.render(title=title)




