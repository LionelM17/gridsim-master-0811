from Agent.DDPGAgent import DDPG_Agent
from utils import get_action_space, get_state_from_obs, legalize_action, add_normal_noise, check_extreme_action
from Agent.RandomAgent import RandomAgent
from Agent.PPOAgent import PPO
from utilize.settings import settings
import numpy as np
import copy
from test import run_task
from ReplayBuffer import LinearSchedule

def interact_with_environment(env, replay_buffer, action_dim, state_dim, device, parameters, summary_writer):
    score_old = 0.0
    policy_agent = DDPG_Agent(settings, replay_buffer, device, action_dim, state_dim, parameters)
    rand_agent = RandomAgent(settings.num_gen)
    obs, done = env.reset(), False
    state = get_state_from_obs(obs, settings)
    action_high, action_low = get_action_space(obs, parameters, settings)
    episode_start = True
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    eps = policy_agent.initial_eps

    std_noise_schedule = LinearSchedule(parameters['max_timestep'], final_p=0.0, initial_p=0.3)

    # interact with the enviroment for max_timesteps
    info_train = None
    # print('no noise')
    for t in range(parameters['max_timestep']):
        episode_timesteps += 1

        if parameters['random_explore'] == 'EpsGreedy':
            # greedy eps
            if np.random.uniform(0, 1) < eps:
                action = rand_agent.act(obs)
                # action = policy_agent.act(state, obs)
            else:
                action = policy_agent.act(state, obs)
                check_extreme_action(action, action_high, action_low, settings, parameters['only_power'], parameters['only_thermal'])
        elif parameters['random_explore'] == 'Gaussian':
            # Gaussian Noise
            action = policy_agent.act(state, obs)
            check_extreme_action(action, action_high, action_low, settings, parameters['only_power'], parameters['only_thermal'])
            noise_std = std_noise_schedule.value(t)
            action = add_normal_noise(noise_std, action, action_high, action_low, settings, parameters['only_power'], parameters['only_thermal'])   # add normal noise on action to improve exploration
        elif parameters['random_explore'] == 'none':
            action = policy_agent.act(state, obs)
            check_extreme_action(action, action_high, action_low, settings, parameters['only_power'],
                                 parameters['only_thermal'])

        # env step
        next_obs, reward, done, info = env.step(action)
        # print('renewable_gen_adj:', np.asarray(action['adjust_gen_p'])[settings.renewable_ids])
        # print(np.asarray(next_obs.gen_p)[settings.renewable_ids] - np.asarray(next_obs.curstep_renewable_gen_p_max))
        # import ipdb
        # ipdb.set_trace()
        if sum(np.asarray(next_obs.gen_p)[settings.renewable_ids]) > sum(next_obs.load_p):
            print('renewable power is more ...')
        next_state = get_state_from_obs(next_obs, settings)
        next_action_high, next_action_low = get_action_space(next_obs, parameters, settings)

        # Train agent after collecting sufficient data
        if t >= parameters["start_timesteps"] and t % parameters['train_freq'] == 0:
            info_train = policy_agent.train(t)  # replay buffer sampled state mismatch obs ?

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
            print(info)
            # if episode_num % 10:
            print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            if episode_num % 20 == 0:
                summary_writer.add_scalar('episode/cumulative reward', episode_reward, t)
                summary_writer.add_scalar('episode/average_reward', episode_reward / episode_timesteps, t)
                summary_writer.add_scalar('episode/total_steps', episode_timesteps, t)
                if parameters['random_explore'] == 'EpsGreedy':
                    summary_writer.add_scalar('statistics/epsilon', eps, t)
                elif parameters['random_explore'] == 'Gaussian':
                    summary_writer.add_scalar('statistics/std', noise_std, t)
                if info_train is not None:
                    for k, v in info_train.items():
                        summary_writer.add_scalar(k, v, t)
            # summary.add_scalar('P_loss', P_loss, t)
            # Reset environment
            obs, done = env.reset(), False
            state = get_state_from_obs(obs, settings)
            action_high, action_low = get_action_space(obs, parameters, settings)
            episode_start = True
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            if eps > policy_agent.end_eps:
                eps *= policy_agent.eps_decay
            else:
                eps = policy_agent.end_eps
            print(f'epsilon={eps:.3f}')

        # if t > 0 and t % parameters["model_save_interval"] == 0:
        #     policy_agent.save(f'./models/model_{t}')

        if t % parameters["test_interval"] == 0:
            mean_score = run_task(policy_agent)
            print(f'-------------test_mean_score = {mean_score} -----------------')
            if mean_score > score_old:
                policy_agent.save(f'./models_best/model_2_')
                score_old = mean_score
            summary_writer.add_scalar('test/episodic_mean_score', mean_score, t)
    return policy_agent


def PPO_train(env, action_dim, state_dim, device, parameters, summary_writer, settings):
    policy_agent = PPO(state_dim, action_dim, parameters, device, parameters['lr_actor'], parameters['lr_critic'],
                        parameters['gamma'], parameters['K_epochs'], parameters['eps_clip'], settings, parameters['continuous_action_space'], parameters['action_std'])

    episode_reward = 0
    episode_num = 0

    # for t in range(parameters['max_timestep']):
    t = 0
    while t < parameters['max_timestep']:
        episode_num += 1
        obs, done = env.reset(), False
        state = get_state_from_obs(obs, settings)
        episode_reward = 0
        episode_timesteps = 0

        for ep in range(parameters['max_episode']):
            episode_timesteps += 1
            action = policy_agent.select_action(state, obs)
            next_obs, reward, done, info = env.step(action)
            next_state = get_state_from_obs(next_obs, settings)

            # saving reward and is_terminals
            policy_agent.buffer.rewards.append(reward)
            policy_agent.buffer.is_terminals.append(done)
            episode_reward += reward

            t += 1
            # update obs and state
            state = copy.copy(next_state)
            obs = copy.copy(next_obs)

            # update PPO agent
            if t % parameters['train_freq'] == 0:
                policy_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if parameters['continuous_action_space'] and t % parameters['action_std_decay_freq'] == 0:
                policy_agent.decay_action_std(parameters['action_std_decay_rate'], parameters['min_action_std'])

            # save model weights
            if t > 0 and t % parameters["model_save_interval"] == 0:
                policy_agent.save(f'./models/model_{t}')

            if done:
                if episode_num % 10 == 0:
                    print(info)
                    print(
                        f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                if episode_num % 200 == 0:
                    summary_writer.add_scalar('reward', episode_reward, t)
                    summary_writer.add_scalar('episode_timesteps', episode_timesteps, t)
                    # if info_train is not None:
                    #     for k, v in info_train.items():
                    #         summary_writer.add_scalar(k, v, t)
                break

    return policy_agent