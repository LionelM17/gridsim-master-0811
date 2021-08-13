# -*- coding: UTF-8 -*-
import numpy as np
import torch
import copy

from Agent.DoNothingAgent import DoNothingAgent
from Agent.RandomAgent import RandomAgent
from Environment.base_env import Environment
from utilize.settings import settings
import DDPG

def run_task(my_agent):

    for episode in range(max_episode):
        print('------ episode ', episode)
        env = Environment(settings, "EPRIReward")
        print('------ reset ')
        obs = env.reset()

        reward = 0.0
        done = False
        
        # while not done:
        for timestep in range(max_timestep):
            ids = [i for i,x in enumerate(obs.rho) if x > 1.0]
            # print("overflow rho: ", [obs.rho[i] for i in ids])    
            print('------ step ', timestep)
            action = my_agent.act(obs, reward, done)
            # print("adjust_gen_p: ", action['adjust_gen_p'])
            # print("adjust_gen_v: ", action['adjust_gen_v'])
            obs, reward, done, info = env.step(action)
            print('info:', info)
            if done:
                break

def interact_with_environment(env, replay_buffer, action_dim, state_dim, device, parameters):
    policy_agent = DDPG.DDPG_Agent(settings.num_gen, action_dim, state_dim)

    state, done = env.reset(), False
    episode_start = True
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    eps = policy_agent.initial_eps

    # interact with the enviroment for max_timesteps
    for t in range(parameters.max_timesteps):
        episode_timesteps += 1
        if t < parameters["start_timesteps"]:
            action = RandomAgent.act(state)
        else:
            action =  policy_agent.select_action(np.array(state), eps)
        #print(action)
        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        done_float = float(done)

        # add to replaybuffer
        replay_buffer.add(state, action, next_state, reward, done_float, done, episode_start)
        state = copy.copy(next_state)
        episode_start = False

        # Train agent after collecting sufficient data
        if t >= parameters["start_timesteps"] and (t+1) % parameters["train_freq"] == 0:
            info = policy_agent.train(replay_buffer)
            # for k,v in info.items():
            #     summary.add_scalar(k, v, t)

        if (t % 16 == 0):
            policy_agent.copy_target_update()
        # input_data = input[t:t + 200]
        # policy, episode_reward = MPC_control(t, env, input_data, parameters["control_circle"], policy, state,
        #                                      replay_buffer, 5)
        # action = policy.select_action(np.array(state), eps)
        # next_state, reward, done, P_loss = env.step(action)
        # state = copy.copy(next_state)
        # info = policy.train(replay_buffer)
        # for k, v in info.items():
        #     summary.add_scalar(k, v, t)
        episode_start = False

        if done:
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} average_Reward:{episode_reward / episode_timesteps:.3f}")
            summary.add_scalar('reward', episode_reward, t)
            summary.add_scalar('average_reward', episode_reward / episode_timesteps, t)
            summary.add_scalar('episode_timesteps', episode_timesteps, t)
            # summary.add_scalar('P_loss', P_loss, t)
            # Reset environment
            state, done = env.reset(), False
            episode_start = True
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            if eps > policy_agent.end_eps:
                eps *= policy_agent.eps_decay
            else:
                eps = policy_agent.end_eps
            #print(eps)

    return policy_agent

if __name__ == "__main__":
    # max_timestep = 10  # 最大时间步数
    # max_episode = 1  # 回合数
    #
    # my_agent = RandomAgent(settings.num_gen)
    #
    # run_task(my_agent)
    parameters = {
        "start_timesteps": 2,
        "initial_eps": 0.9,
        "end_eps": 0.001,
        "eps_decay": 0.99,
        # Evaluation
        "eval_freq": int(5e2),
        # Learning
        "gamma": 0.99,
        "batch_size": 16,
        "optimizer": "Adam",
        "optimizer_parameters": {
            "lr": 5e-3
        },
        "train_freq": 1,
        "target_update_fre": 1,
        "tau": 0.001,
        "control_circle": 3,
        "input_size": 10,
        "hidden_size": 35,
        "num_layers": 3,
        "seq_len": 3,
        "lstm_batch_size": 1,
        "output_size": 1,
        "max_timestep": 10,
        "max_episode": 1,
        "buffer_size": 1e6
    }
    # get state dim and action dim
    state_dim = 10

    env = Environment(settings, "EPRIReward")
    obs = env.reset()
    action_dim_p = obs.action_space['adjust_gen_p'].size()
    action_dim_v = obs.action_space['adjust_gen_v'].size()
    action_dim = action_dim_p + action_dim_v

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    replay_buffer = DDPG.StandardBuffer(state_dim, action_dim, parameters["batch_size"], parameters["buffer_size"], device)
    trained_policy_agent = interact_with_environment(env, replay_buffer, action_dim, state_dim, device, parameters)
