import math
import torch
import os
import pandas as pd

from envs.forward import DiffEnv
from envs.discretization import euler, runge_kutta4, runge_kutta4_cst_ctrl
from envs.utils.tracks import get_track
from envs.utils.torch_utils import smooth_relu
from envs.utils.car_visualization import plot_track, plot_traj


class Car(DiffEnv):
    def __init__(self, dt=0.02, discretization='rk4_cst_ctrl', horizon=20,
                 track='simple', model='simple',
                 reg_ctrl=1e-6, reg_cont=0.1, reg_lag=10., reg_speed=0.1, reg_bar=100., reg_obs=0.,
                 cost='contouring', time_penalty='vref_squared',
                 vref=3., vinit=1., constrain_angle=math.pi/3, acc_bounds=(-0.1, 1.), seed=0
                 ):
        super(Car, self).__init__()
        if seed != 0:
            torch.manual_seed(seed)
            vinit = vinit + 1e-2*torch.randn(1).item()
            reg_cont, reg_lag, reg_speed = reg_cont + 1e-3 * torch.randn(1).item(), \
                                           reg_lag + 1e-3 * torch.randn(1).item(), \
                                           reg_speed + 1e-3 * torch.randn(1).item()

        # dyn parameters
        self.dt, self.discretization, self.horizon = dt, discretization, horizon
        self.model = model
        self.vref, self.constrain_angle, self.acc_bounds = vref, constrain_angle, acc_bounds

        # cost parameters
        self.reg_ctrl, self.reg_cont, self.reg_lag, self.reg_speed, self.reg_barr, self.reg_obs = \
            reg_ctrl, reg_cont, reg_lag, reg_speed, reg_bar, reg_obs
        self.cost_type, self.time_penalty = cost, time_penalty

        # track parameters
        self.track, self.inner_track, self.outer_track = get_track(track)
        dx_ref, dy_ref = self.track.derivative(torch.tensor(0.))
        phi_ref = torch.atan2(dy_ref, dx_ref)

        if model == 'simple':
            self.init_state = torch.tensor([0., 0., phi_ref, vinit, 0., vinit])
            self.dim_ctrl, self.dim_state = 3, 6
        elif model == 'real':
            self.init_state = torch.tensor([0., 0., phi_ref, vinit, 0., 0., 0., vinit])
            self.dim_ctrl, self.dim_state = 3, 8
        else:
            raise NotImplementedError

        if self.discretization == 'RK4':
            self.dim_ctrl *= 3

        dir_name = os.path.dirname(os.path.abspath(__file__))
        self.dyn_csts = pd.read_json(os.path.join(dir_name, 'utils/model.json'), typ='series')

        self.cartrans = None
        self.init_time_iter = 0

        time_obstacle = max(self.track._t) / 20
        self.obstacle = self.track.evaluate(time_obstacle) + torch.tensor([0.2, 0.])

    def discrete_dyn(self, state, ctrl, time_iter):
        if self.model == 'real':
            dyn = self.bicycle_model
        elif self.model == 'simple':
            dyn = self.simple_model
        else:
            raise NotImplementedError
        if self.discretization == 'euler':
            next_state = euler(dyn, state, ctrl, self.dt)
        elif self.discretization == 'rk4':
            next_state = runge_kutta4(dyn, state, ctrl, self.dt)
        elif self.discretization == 'rk4_cst_ctrl':
            next_state = runge_kutta4_cst_ctrl(dyn, state, ctrl, self.dt)
        else:
            raise NotImplementedError
        return next_state

    def simple_model(self, state, ctrl):
        x, y, phi, v, tref, vtref = state
        a, delta, atref = ctrl
        delta = 2/math.pi*torch.arctan(delta)*self.constrain_angle
        carlength = 1.5*self.dyn_csts['car_l']

        dx = v * torch.cos(phi)
        dy = v * torch.sin(phi)
        dv = a
        dphi = v * torch.tan(delta)/carlength
        dtref = vtref
        dvtref = atref
        return torch.stack([dx, dy, dphi, dv, dtref, dvtref])

    def bicycle_model(self, state, ctrl):
        x, y, phi, vx, vy, vphi, tref, vtref = state
        a, delta, atref = ctrl
        delta = 2/math.pi*torch.arctan(delta)*self.constrain_angle
        # hardcode constraints on the acceleration
        gap = self.acc_bounds[1] - self.acc_bounds[0]
        a = gap*torch.sigmoid(4 * a / gap) + self.acc_bounds[0]


        Cm1, Cm2, Cr0, Cr2,\
        Br, Cr, Dr, Bf, Cf, Df,\
        m, Iz, lf, lr = [self.dyn_csts[key] for key in ['Cm1', 'Cm2', 'Cr0', 'Cr2',
                                                        'Br', 'Cr', 'Dr', 'Bf', 'Cf', 'Df',
                                                        'm', 'Iz', 'lf', 'lr']]
        alphaf = delta - torch.atan2(vphi*lf + vy, vx)
        alphar = torch.atan2(vphi*lr - vy, vx)
        Fry = Dr*torch.sin(Cr*torch.atan(Br*alphar))
        Ffy = Df*torch.sin(Cf*torch.atan(Bf*alphaf))
        Frx = (Cm1 - Cm2*vx)*a - Cr0 - Cr2*vx**2
        dx = torch.cos(phi)*vx - torch.sin(phi)*vy
        dy = torch.sin(phi)*vx + torch.cos(phi)*vy
        dphi = vphi
        dvx = (Frx - Ffy*torch.sin(delta) + m*vy*vphi)/m
        dvy = (Fry + Ffy*torch.cos(delta) - m*vx*vphi)/m
        dvphi = (Ffy*lf*torch.cos(delta) - Fry*lr)/Iz
        dtref = vtref
        dvtref = atref
        return torch.stack([dx, dy, dphi, dvx, dvy, dvphi, dtref, dvtref])

    def costs(self, next_state, ctrl, time_iter):
        if self.model == 'simple':
            x, y, phi, v, tref, vtref = next_state
        else:
            x, y, phi, vx, vy, vphi, tref, vtref = next_state

        if self.cost_type == 'contouring':
            qC, qL, qVs = self.reg_cont, self.reg_lag, self.reg_speed
            x_ref, y_ref = self.track.evaluate(tref)
            dx_ref, dy_ref = self.track.derivative(tref)
            phi_ref = torch.atan2(dy_ref, dx_ref)

            cont_err = torch.sin(phi_ref)*(x-x_ref) - torch.cos(phi_ref)*(y-y_ref)
            lag_err = -torch.cos(phi_ref)*(x-x_ref) - torch.sin(phi_ref)*(y-y_ref)
            cost_next_state = qC*cont_err**2 + qL*lag_err**2
            if self.time_penalty == 'tref':
                cost_next_state = cost_next_state - qVs*tref
            elif self.time_penalty == 'dref':
                cost_next_state = cost_next_state - qVs*vtref*self.dt
            elif self.time_penalty == 'dref_squared':
                cost_next_state = cost_next_state - qVs*vtref*self.dt*tref
            elif self.time_penalty == 'vref_squared':
                cost_next_state = cost_next_state + qVs*(vtref-self.vref)**2*self.dt**2
            else:
                raise NotImplementedError
        elif self.cost_type == 'exact':
            time = torch.tensor(time_iter*self.vref*self.dt)
            x_ref, y_ref = self.track.evaluate(time)
            cost_next_state = (x - x_ref)**2 + (y - y_ref)**2
        else:
            raise NotImplementedError

        if type(self.reg_ctrl) == tuple:
            if self.discretization in ['euler', 'RK4_cst_ctrl']:
                a, delta, atref = ctrl
                cost_ctrl = self.reg_ctrl[0]*(a**2 + delta**2) + self.reg_ctrl[1]*atref**2
            elif self.discretization == 'RK4':
                accs, deltas, atrefs = ctrl[::3], ctrl[1::3], ctrl[2::3]
                cost_ctrl = self.reg_ctrl[0]*((accs**2).sum() + (deltas**2).sum()) \
                            + self.reg_ctrl[1]*(atrefs**2).sum()
            else:
                raise NotImplementedError
        else:
            cost_ctrl = self.reg_ctrl*ctrl.dot(ctrl)

        barr_next_state = self.barrier(next_state)
        cost_next_state = cost_next_state + barr_next_state

        return cost_next_state, cost_ctrl

    def barrier(self, next_state):
        barr_next_state = -1e-6*torch.log(next_state[-1])

        carwidth = 1.5*self.dyn_csts['car_w']

        tref = next_state[-2]
        x, y = next_state[:2]
        point = torch.tensor([x, y])
        for i, border in enumerate([self.inner_track, self.outer_track]):
            border_point = border.evaluate(tref)
            dpos = border.derivative(tref)
            v = torch.sqrt(torch.sum(dpos**2))
            normal = torch.tensor([dpos[1], -dpos[0]])/v
            if i == 0:
                # inner
                barr_next_state = barr_next_state \
                                  + self.reg_barr * smooth_relu(carwidth / 2 - (point - border_point).dot(normal)) ** 2
            else:
                # outer
                barr_next_state = barr_next_state \
                                  + self.reg_barr * smooth_relu((point - border_point).dot(normal) + carwidth / 2) ** 2
        barr_next_state = barr_next_state

        if self.reg_obs > 0.:
            barr_next_state = barr_next_state \
                              + self.reg_obs * smooth_relu(carwidth ** 2 - torch.sum((point - self.obstacle) ** 2))

        return barr_next_state

    def set_viewer(self):
        from envs.utils.car_visualization import set_window_viewer, add_car_to_viewer, \
            add_track_to_viewer, add_obstacle_to_viewer
        nb_points = 500
        carlength, carwidth = 1.5 * self.dyn_csts['car_l'], 1.5 * self.dyn_csts['car_w']

        time = torch.linspace(0, max(self.track._t), nb_points)
        track_points = self.outer_track.evaluate(time).numpy()
        min_x, max_x = min(track_points[:, 0]), max(track_points[:, 0])
        min_y, max_y = min(track_points[:, 1]), max(track_points[:, 1])

        self.viewer = set_window_viewer(min_x, max_x, min_y, max_y, 2*carlength)
        add_track_to_viewer(self.viewer, self.track, self.inner_track, self.outer_track)
        if self.reg_obs > 0.:
            add_obstacle_to_viewer(self.viewer, self.obstacle.numpy(), carlength)

        self.cartrans = add_car_to_viewer(self.viewer, carlength, carwidth)

    def plot_track(self, fig=None, ax=None):
        fig, ax = plot_track(self.track, self.inner_track, self.outer_track, fig=fig, ax=ax)
        return fig, ax

    def plot_traj(self, traj, fig, ax, add_colorbar=True):
        return plot_traj(traj, fig, ax, model=self.model, add_colorbar=add_colorbar)

    def render(self, title=None):
        if self.viewer is None:
            self.set_viewer()
        self.cartrans.set_translation(self.state[0].item(), self.state[1].item())
        self.cartrans.set_rotation(self.state[2].item())
        return self.viewer.render(title=title)




