# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import copy

from Agent.RandomAgent import RandomAgent
from Environment.base_env import Environment
from utilize.settings import settings
from ReplayBuffer import StandardBuffer
from Agent.DDPGAgent import DDPG_Agent
from utils import get_action_space, get_state_from_obs, legalize_action, add_normal_noise, check_extreme_action
import json
# import ray
import warnings
warnings.filterwarnings('ignore')

def run_task(my_agent):
    max_episode = 10
    max_timestep = 10000
    episode_reward = [0 for _ in range(max_episode)]
    for episode in range(max_episode):
        print('------ episode ', episode)
        env = Environment(settings, "EPRIReward")
        print('------ reset ')
        obs = env.reset()

        reward = 0.0
        done = False
        
        # while not done:
        for timestep in range(max_timestep):
            ids = [i for i, x in enumerate(obs.rho) if x > 1.0]
            print("overflow rho: ", [obs.rho[i] for i in ids])
            print('------ step ', timestep)
            state = get_state_from_obs(obs, settings)
            action = my_agent.act(state, obs)
            obs, reward, done, info = env.step(action)
            episode_reward[episode] += reward
            if done:
                print('info:', info)
                print(f'episode cumulative reward={episode_reward[episode]}')
                break

        return sum(episode_reward) / len(episode_reward)

def interact_with_environment(env, replay_buffer, action_dim, state_dim, device, parameters, summary_writer):
    policy_agent = DDPG_Agent(settings, replay_buffer, device, action_dim, state_dim, parameters)
    rand_agent = RandomAgent(settings.num_gen)
    obs, done = env.reset(), False
    state = get_state_from_obs(obs, settings)
    action_high, action_low = get_action_space(obs, parameters)
    episode_start = True
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    eps = policy_agent.initial_eps

    # interact with the enviroment for max_timesteps
    info_train = None
    # print('no noise')
    for t in range(parameters['max_timestep']):
        episode_timesteps += 1

        if parameters['random_explore'] == 'EpsGreedy':
            # greedy eps
            if np.random.uniform(0, 1) < eps:
                action = rand_agent.act(obs)
            else:
                action = policy_agent.act(state, obs)
                check_extreme_action(action, action_high, action_low, parameters['only_power'])
        elif parameters['random_explore'] == 'Gaussian':
            # Gaussian Noise
            action = policy_agent.act(state, obs)
            check_extreme_action(action, action_high, action_low, parameters['only_power'])
            action = add_normal_noise(action, action_high, action_low, parameters['only_power'])   # add normal noise on action to improve exploration

        # env step
        next_obs, reward, done, info = env.step(action)
        next_state = get_state_from_obs(next_obs, settings)
        next_action_high, next_action_low = get_action_space(next_obs, parameters)

        # Train agent after collecting sufficient data
        if t >= parameters["start_timesteps"]:
            info_train = policy_agent.train()  # replay buffer sampled state mismatch obs ?

        if (t % parameters["target_update_interval"] == 0):
            policy_agent.copy_target_update()
        episode_start = False

        episode_reward += reward
        # add to replaybuffer
        replay_buffer.add(state, action, action_high, action_low, next_state, next_action_high, next_action_low, reward, done, episode_start)
        state = copy.copy(next_state)
        obs = copy.copy(next_obs)
        action_high, action_low = copy.copy(next_action_high), copy.copy(next_action_low)

        if done:
            if episode_num % 10:
                print(info)
                print(
                    f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            if episode_num % 200 == 0:
                summary_writer.add_scalar('reward', episode_reward, t)
                summary_writer.add_scalar('average_reward', episode_reward / episode_timesteps, t)
                summary_writer.add_scalar('episode_timesteps', episode_timesteps, t)
                if info_train is not None:
                    for k, v in info_train.items():
                        summary_writer.add_scalar(k, v, t)
            # summary.add_scalar('P_loss', P_loss, t)
            # Reset environment
            obs, done = env.reset(), False
            state = get_state_from_obs(obs, settings)
            action_high, action_low = get_action_space(obs, parameters)
            episode_start = True
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            if eps > policy_agent.end_eps:
                eps *= policy_agent.eps_decay
            else:
                eps = policy_agent.end_eps
            # print(f'epsilon={eps:.3f}')

        if t > 0 and t % parameters["model_save_interval"] == 0:
            policy_agent.save(f'./models/model_{t}')

        if t % parameters["test_interval"] == 0:
            mean_score = run_task(policy_agent)
            summary_writer.add_scalar('test_mean_score', mean_score, t)
    return policy_agent

if __name__ == "__main__":
    parameters = {
        "start_timesteps": 100,
        "initial_eps": 0.9,
        "end_eps": 0.001,
        "eps_decay": 0.999,
        # Evaluation
        "eval_freq": int(5e2),
        # Learning
        "gamma": 0.99,
        "batch_size": 32,
        "optimizer": "Adam",
        "optimizer_parameters": {
            "lr": 0.001
        },
        "train_freq": 150,
        "target_update_fre": 16,
        "tau": 0.001,
        "control_circle": 3,
        "input_size": 10,
        "hidden_size": 35,
        "num_layers": 3,
        "seq_len": 3,
        "lstm_batch_size": 1,
        "output_size": 1,
        "max_timestep": 10000000,
        "max_episode": 50000,
        "buffer_size": 1000 * 1000,
        "target_update_interval": 300,
        "model_save_interval": 10000,
        "test_interval": 1000,
        "only_power": True,
        "padding_state": False,
        "random_explore": 'EpsGreedy'  # Gaussian or EpsGreedy
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
    assert action_dim_v == action_dim_p
    if parameters['only_power']:
        action_dim = action_dim_p
    else:
        action_dim = action_dim_p + action_dim_v

    state = get_state_from_obs(obs, settings)
    state_dim = len(state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    replay_buffer = StandardBuffer(state_dim, action_dim, parameters, device)
    trained_policy_agent = interact_with_environment(env, replay_buffer, action_dim, state_dim, device, parameters, summary_writer)
    run_task(trained_policy_agent)
