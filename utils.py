import numpy as np
from sklearn import preprocessing
import torch
from utilize.form_action import *

def get_state_from_obs(obs, settings):
    # state_form = {'gen_p', 'gen_q', 'gen_v', 'load_p', 'load_q', 'load_v', 'p_or', 'q_or', 'v_or', 'a_or', 'p_ex',
    #               'q_ex', 'v_ex', 'a_ex',
    #               'rho', 'grid_loss', 'curstep_renewable_gen_p_max'}
    state_form = {'gen_p', 'gen_q', 'gen_v', 'target_dispatch', 'actual_dispatch', 'load_p', 'load_q', 'load_v', 'rho'}
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

def get_action_space(obs, parameters, settings):
    if parameters['only_power']:
        if parameters['only_thermal']:
            action_high = obs.action_space['adjust_gen_p'].high[settings.thermal_ids]
            action_low = obs.action_space['adjust_gen_p'].low[settings.thermal_ids]
        else:
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

def add_normal_noise(std, action, action_high, action_low, settings, only_power=False, only_thermal=False):
    if only_power:
        if only_thermal:
            action_dim = len(settings.thermal_ids)
            err_p_thermal = np.random.normal(0, std, action_dim)
            action['adjust_gen_p'][settings.thermal_ids] += err_p_thermal
            action['adjust_gen_p'][settings.thermal_ids] = action['adjust_gen_p'][settings.thermal_ids].clip(action_low, action_high)
        else:
            action_dim = len(action['adjust_gen_p'])
            err_p = np.random.normal(0, std, action_dim)
            action['adjust_gen_p'] += err_p
            action['adjust_gen_p'] = action['adjust_gen_p'].clip(action_low, action_high)
    else:
        action_dim = len(action['adjust_gen_p']) + len(action['adjust_gen_v'])
        err_p, err_v = np.random.normal(0, std, action_dim // 2), np.random.normal(0, std / 10, action_dim // 2)
        # print(f'max_err_p={max(abs(err_p)):.3f}, max_err_v={max(abs(err_v)):.3f}')
        action['adjust_gen_p'] += err_p
        action['adjust_gen_p'] = action['adjust_gen_p'].clip(action_low[:action_dim // 2], action_high[:action_dim // 2])
        action['adjust_gen_v'] += err_v
        action['adjust_gen_v'] = action['adjust_gen_v'].clip(action_low[action_dim // 2:], action_high[action_dim // 2:])

    return action

def check_extreme_action(action, action_high, action_low, settings, only_power=False, only_thermal=False):
    if only_power:
        if only_thermal:
            action = np.asarray(action['adjust_gen_p'])[settings.thermal_ids]
        else:
            action = action['adjust_gen_p']
    else:
        action = np.asarray([action['adjust_gen_p'], action['adjust_gen_v']]).flatten()

    assert len(action) == len(action_high) == len(action_low)
    cnt = 0
    for i in range(len(action)):
        if action[i] == action_high[i] or action[i] == action_low[i]:
            if action_high[i] == action_low[i]:
                continue
            print(f'action={action[i]}, high={action_high[i]}, low={action_low[i]}')
            cnt += 1

    # print(f'extreme action percentage={cnt/len(action):.3f}')

def voltage_action(obs, settings, type='1'):  # type = 'period' or '1'
    if type == 'period':
        err = np.maximum(np.asarray(settings.min_gen_v) - np.asarray(obs.gen_v), np.asarray(obs.gen_v) - np.asarray(settings.max_gen_v))  # restrict in legal range
        gen_num = len(err)
        action_high, action_low = obs.action_space['adjust_gen_v'].high, obs.action_space['adjust_gen_v'].low
        adjust_gen_v = np.zeros(54)
        for i in range(gen_num):
            if err[i] <= 0:
                continue
            elif err[i] > 0:
                if obs.gen_v[i] < settings.min_gen_v[i]:
                    if err[i] < action_high[i]:
                        adjust_gen_v[i] = err[i]
                    else:
                        adjust_gen_v[i] = action_high[i]
                elif obs.gen_v[i] > settings.max_gen_v[i]:
                    if - err[i] > action_low[i]:
                        adjust_gen_v[i] = - err[i]
                    else:
                        adjust_gen_v[i] = action_low[i]
    elif type == '1':
        # err = obs.gen_v - (np.asarray(settings.max_gen_v) + np.asarray(settings.min_gen_v)) / 2  # restrict at stable point 1
        err = np.asarray(obs.gen_v) - 1.05  # restrict at stable point 1.05
        gen_num = len(err)
        action_high, action_low = obs.action_space['adjust_gen_v'].high, obs.action_space['adjust_gen_v'].low
        adjust_gen_v = np.zeros(54)
        for i in range(gen_num):
            if err[i] < 0:
                if - err[i] > action_high[i]:
                    adjust_gen_v[i] = action_high[i]
                else:
                    adjust_gen_v[i] = - err[i]
            elif err[i] > 0:
                if - err[i] < action_low[i]:
                    adjust_gen_v[i] = action_low[i]
                else:
                    adjust_gen_v[i] = - err[i]

    return adjust_gen_v

'''
set renewable gen_p to max
'''
def adjust_renewable_generator(obs, settings):
    adjust_gen_renewable = obs.action_space['adjust_gen_p'].high[settings.renewable_ids]
    # gen_renewable = obs.gen_p[settings.renewable_ids] + adjust_gen_renewable
    # if sum(obs.gen_p[settings.renewable_ids]) > sum(obs.nextstep_load_p):
    #
    return adjust_gen_renewable

def form_p_action(adjust_gen_thermal, adjust_gen_renewable, settings):
    thermal_dims = len(adjust_gen_thermal)
    thermal_ids = settings.thermal_ids
    renewable_dims = len(adjust_gen_renewable)
    renewable_ids = settings.renewable_ids
    adjust_gen_p = np.zeros(54)
    for i in range(thermal_dims):
        adjust_gen_p[thermal_ids[i]] = adjust_gen_thermal[i]
    for i in range(renewable_dims):
        adjust_gen_p[renewable_ids[i]] = adjust_gen_renewable[i]
    return adjust_gen_p