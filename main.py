# -*- coding: UTF-8 -*-
import torch
from torch.utils.tensorboard import SummaryWriter
from Environment.base_env import Environment
from utilize.settings import settings
from ReplayBuffer import StandardBuffer
from utils import get_state_from_obs
from train import interact_with_environment, PPO_train
from test import run_task

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    parameters = {
        "start_timesteps": 500,
        "initial_eps": 0.9,
        "end_eps": 0.001,
        "eps_decay": 0.999,
        # Learning
        "gamma": 0.997,
        "batch_size": 512,
        "optimizer": "Adam",
        "optimizer_parameters": {
            "lr": 0.001,
            "weight_decay": 5e-3
        },
        "train_freq": 600,  # PPO paper 4000
        "tau": 0.001,
        "control_circle": 3,
        "seq_len": 3,
        "lstm_batch_size": 1,
        "output_size": 1,
        "max_timestep": 10000000,
        "max_episode": 1000,
        "buffer_size": 1000 * 1000,
        "target_update_interval": 4000 * 2,
        "model_save_interval": 10000,
        "test_interval": 1000,
        "only_power": True,
        "only_thermal": True,
        # "action_type": 'only_thermal',  # only_power, only_thermal
        "padding_state": False,
        "random_explore": 'EpsGreedy',  # Gaussian or EpsGreedy
        # PPO parameters
        'lr_actor': 0.0003,
        'lr_critic': 0.001,
        'K_epochs': 20,
        'eps_clip': 0.2,
        'action_std': 0.6,
        'action_std_decay_rate': 0.05,
        'min_action_std': 0.1,
        'action_std_decay_freq': int(2.5e5),
        'continuous_action_space': True
    }

    # seed
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    summary_writer = SummaryWriter()
    # get state dim and action dim
    env = Environment(settings, "EPRIReward")
    obs = env.reset()
    action_dim_p = obs.action_space['adjust_gen_p'].shape[0]
    action_dim_v = obs.action_space['adjust_gen_v'].shape[0]
    action_dim_thermal = len(settings.thermal_ids)
    assert action_dim_v == action_dim_p
    if parameters['only_power']:
        action_dim = action_dim_p
        if parameters['only_thermal']:
            action_dim = action_dim_thermal
    else:
        action_dim = action_dim_p + action_dim_v

    state = get_state_from_obs(obs, settings)
    state_dim = len(state)
    # import ipdb
    # ipdb.set_trace()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    replay_buffer = StandardBuffer(state_dim, action_dim, parameters, device, settings)
    trained_policy_agent = interact_with_environment(env, replay_buffer, action_dim, state_dim, device, parameters, summary_writer)
    run_task(trained_policy_agent)

    # trained_policy_agent = PPO_train(env, action_dim, state_dim, device, parameters, summary_writer, settings)
