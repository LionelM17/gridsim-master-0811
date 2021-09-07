# -*- coding: UTF-8 -*-
import torch
from torch.utils.tensorboard import SummaryWriter
from Environment.base_env import Environment
from utilize.settings import settings
from ReplayBuffer import StandardBuffer
from utils import get_state_from_obs
from train import interact_with_environment, PPO_train
from test import run_task
import ray
import copy
from DDPG import ActorNet, CriticNet

from Apex.Actor import Actor_DDPG, Storage, SharedMemory, Testor
from Apex.Learner import Learner_DDPG, SharedBuffer

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    parameters = {
        "start_timesteps": 50,
        # Learning
        "gamma": 0.997,
        "batch_size": 512,
        "optimizer": "Adam",
        "optimizer_parameters": {
            "lr": 0.001,
            "weight_decay": 5e-3
        },
        "tau": 0.001,
        "training_iterations": 1000 * 1000,
        "buffer_size": 1000 * 1000,
        "target_update_interval": 100,
        "test_interval": 5000,
        "only_power": True,
        "only_thermal": True,
        "random_explore": 'Gaussian',  # Gaussian or EpsGreedy or none
        'actor_num': 20,
        'actor_update_interval': 200,
        'log_interval': 1000,
        'total_transitions': 200 * 1000 * 1000
    }

    # seed
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

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
    parameters['action_dim'] = action_dim

    state = get_state_from_obs(obs, settings)
    state_dim = len(state)
    parameters['state_dim'] = state_dim

    ray.init(num_gpus=2, num_cpus=50, object_store_memory=200 * 1024 * 1024 * 1024)

    summary_writer = SummaryWriter()

    actor = ActorNet(state_dim, action_dim, settings)
    actor_target = copy.deepcopy(actor)
    critic = CriticNet(state_dim, action_dim)
    critic_target = copy.deepcopy(critic)

    storage = Storage()
    shared_memory = SharedMemory.remote(actor, actor_target, critic, critic_target)
    shared_buffer = SharedBuffer.remote(parameters, device, settings, storage)
    actors = [Actor_DDPG.remote(i, shared_memory, storage, parameters, settings, device) for i in range(parameters['actor_num'])]

    testor = Testor.remote(settings, device, parameters, shared_memory)

    workers = []
    workers += [shared_buffer.run.remote()]
    workers += [actor.run.remote() for actor in actors]
    workers += [testor.run.remote()]
    learner = Learner_DDPG(0, shared_memory, storage, parameters, settings, device, summary_writer)
    learner.run()
    # learner = Learner_DDPG.remote(0, shared_memory, storage, parameters, settings, device, summary_writer)
    # workers += [learner.run.remote()]

    # ray.wait(workers)

    # trained_policy_agent = PPO_train(env, action_dim, state_dim, device, parameters, summary_writer, settings)
