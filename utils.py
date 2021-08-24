import numpy as np
from sklearn import preprocessing
import torch
from utilize.form_action import *

def get_state_from_obs(obs, settings):
    state_form = {'gen_p', 'gen_q', 'gen_v', 'load_p', 'load_q', 'load_v', 'p_or', 'q_or', 'v_or', 'a_or', 'p_ex',
                  'q_ex', 'v_ex', 'a_ex',
                  'rho', 'grid_loss', 'curstep_renewable_gen_p_max'}
    state = []
    for name in state_form:
        value = getattr(obs, name)
        # if name == 'gen_p':
        #     value = [value[i] / settings.max_gen_p[i] for i in range(len(value))]
        # elif name == 'gen_q':
        #     value = [value[i] / settings.max_gen_q[i] for i in range(len(value))]
        # elif name == 'gen_v':
        #     value = [value[i] / settings.max_gen_v[i] for i in range(len(value))]
        # elif name == 'load_v':
        #     value = [value[i] / settings.max_bus_v[i] for i in range(len(value))]
        # elif name in {'load_p', 'load_q', 'p_or', 'q_or', 'v_or', 'a_or', 'p_ex',
        #               'q_ex', 'v_ex', 'a_ex', 'rho'}:
        value = preprocessing.minmax_scale(np.array(value, dtype=np.float32), feature_range=(0, 1), axis=0, copy=True)
        state.append(np.reshape(np.array(value, dtype=np.float32), (-1,)))
    state = np.concatenate(state)
    return state

def get_action_space(obs, parameters):
    if parameters['only_power']:
        action_high = obs.action_space['adjust_gen_p'].high
        action_high = np.where(np.isinf(action_high), np.full_like(action_high, 0),
                               action_high)  # feed 0 to balanced generator threshold
        action_low = obs.action_space['adjust_gen_p'].low
        action_low = np.where(np.isinf(action_low), np.full_like(action_low, 0),
                              action_low)  # feed 0 to balanced generator threshold
    else:
        action_high = np.asarray(
            [obs.action_space['adjust_gen_p'].high, obs.action_space['adjust_gen_v'].high]).flatten()
        action_high = np.where(np.isinf(action_high), np.full_like(action_high, 0),
                               action_high)  # feed 0 to balanced generator threshold
        action_low = np.asarray([obs.action_space['adjust_gen_p'].low, obs.action_space['adjust_gen_v'].low]).flatten()
        action_low = np.where(np.isinf(action_low), np.full_like(action_low, 0),
                              action_low)  # feed 0 to balanced generator threshold
    return action_high, action_low

def legalize_action(action, settings, obs):

    adjust_gen_p, adjust_gen_v = action
    if len(adjust_gen_p.shape) > 1:
        action_dim = adjust_gen_p.shape[1]
        # TODO: follow siyang
        for i in range(action_dim):
            adjust_gen_p[:, i] = (adjust_gen_p[:, i] + 1) / 2 \
                                 * (obs.action_space['adjust_gen_p'].high[i] - obs.action_space['adjust_gen_p'].low[i]) \
                                 + obs.action_space['adjust_gen_p'].low[i]
    else:
        action_dim = adjust_gen_p.shape[0]
        for i in range(action_dim):
            adjust_gen_p[i] = (adjust_gen_p[i] + 1) / 2 \
                                 * (obs.action_space['adjust_gen_p'].high[i] - obs.action_space['adjust_gen_p'].low[i]) \
                                 + obs.action_space['adjust_gen_p'].low[i]

    return form_action(adjust_gen_p, adjust_gen_v)

def add_normal_noise(action, action_high, action_low, only_power=False):
    if only_power:
        action_dim = len(action['adjust_gen_p'])
        err_p = np.random.normal(0, 0.1, action_dim)
        action['adjust_gen_p'] += err_p
        action['adjust_gen_p'] = action['adjust_gen_p'].clip(action_low, action_high)
    else:
        action_dim = len(action['adjust_gen_p']) + len(action['adjust_gen_v'])
        err_p, err_v = np.random.normal(0, 0.1, action_dim // 2), np.random.normal(0, 0.01, action_dim // 2)
        # print(f'max_err_p={max(abs(err_p)):.3f}, max_err_v={max(abs(err_v)):.3f}')
        action['adjust_gen_p'] += err_p
        action['adjust_gen_p'] = action['adjust_gen_p'].clip(action_low[:action_dim // 2], action_high[:action_dim // 2])
        action['adjust_gen_v'] += err_v
        action['adjust_gen_v'] = action['adjust_gen_v'].clip(action_low[action_dim // 2:], action_high[action_dim // 2:])

    return action

def check_extreme_action(action, action_high, action_low, only_power=False):
    if only_power:
        action = action['adjust_gen_p']
    else:
        action = np.asarray([action['adjust_gen_p'], action['adjust_gen_v']]).flatten()

    assert len(action) == len(action_high) == len(action_low)
    cnt = 0
    for i in range(len(action)):
        if action[i] == action_high[i] or action[i] == action_low[i]:
            cnt += 1

    print(f'extreme action percentage={cnt/len(action):.3f}')