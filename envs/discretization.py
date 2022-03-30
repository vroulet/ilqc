def euler(dyn, state, ctrl, dt):
    return state + dt*dyn(state, ctrl)


def runge_kutta4(dyn, state, ctrl, dt):
    base_dim_ctrl = int(ctrl.shape[0]/3)
    k1 = dyn(state, ctrl[:base_dim_ctrl])
    k2 = dyn(state + dt*k1/2, ctrl[base_dim_ctrl:2*base_dim_ctrl])
    k3 = dyn(state + dt*k2/2, ctrl[base_dim_ctrl:2*base_dim_ctrl])
    k4 = dyn(state + dt*k3, ctrl[2*base_dim_ctrl:])
    return state + dt*(k1 + 2*k2 + 2*k3 + k4)/6


def runge_kutta4_cst_ctrl(dyn, state, ctrl, dt):
    k1 = dyn(state, ctrl)
    k2 = dyn(state + dt*k1/2, ctrl)
    k3 = dyn(state + dt*k2/2, ctrl)
    k4 = dyn(state + dt*k3, ctrl)
    return state + dt*(k1 + 2*k2 + 2*k3 + k4)/6
