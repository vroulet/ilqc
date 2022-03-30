from envs.car import Car
from envs.lin_quad import make_synth_linear_env
from envs.pendulum import Pendulum, CartPendulum


def make_env(env_cfg):
    env = env_cfg['env']
    env_opts = {k: env_cfg[k] for k in set(list(env_cfg.keys())) - {'env'}}
    if env == 'pendulum':
        env = Pendulum(**env_opts)
    elif env == 'cart_pendulum':
        env = CartPendulum(**env_opts)
    elif env == 'real_car':
        env = Car(model='real', **env_opts)
    elif env == 'simple_car':
        env = Car(model='simple', **env_opts)
    elif env == 'synth_linear':
        env = make_synth_linear_env(**env_opts)
    else:
        raise NotImplementedError
    return env

