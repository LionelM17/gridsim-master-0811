import ray
import numpy as np
import torch
from torch.nn import L1Loss
from DDPG import ActorNet, CriticNet
import copy
from Environment.base_env import Environment
from Agent.BaseAgent import BaseAgent
from Agent.RandomAgent import RandomAgent
from ReplayBuffer import LinearSchedule
from utils import get_state_from_obs, check_extreme_action, get_action_space, add_normal_noise, voltage_action, adjust_renewable_generator, form_p_action, form_action
from ray.util.queue import Queue

import time
from utilize.settings import settings

import warnings
warnings.filterwarnings('ignore')

class Storage(object):
    def __init__(self, threshold=15, size=20):
        self.threshold = threshold
        self.batch_queue = Queue(maxsize=size)
        self.trajectory_queue = Queue(maxsize=size)
        self.priority_queue = Queue(maxsize=size)

    def push_batch(self, batch):
        if self.batch_queue.qsize() <= self.threshold:
            self.batch_queue.put(batch)

    def pop_batch(self):
        if self.batch_queue.qsize() > 0:
            return self.batch_queue.get()
        else:
            return None

    def push_trajectory(self, trajectory):
        if self.trajectory_queue.qsize() <= self.threshold:
            self.trajectory_queue.put(trajectory)

    def pop_trajectory(self):
        if self.trajectory_queue.qsize() > 0:
            return self.trajectory_queue.get()
        else:
            return None

    def push_priority(self, priority):
        if self.priority_queue.qsize() <= self.threshold:
            self.priority_queue.put(priority)

    def pop_priority(self):
        if self.priority_queue.qsize() > 0:
            return self.priority_queue.get()
        else:
            return None


@ray.remote
class SharedMemory(object):
    def __init__(self, actor, actor_target, critic, critic_target):
        self.step_counter = 0
        self.actor = actor
        self.actor_target = actor_target
        self.critic = critic
        self.critic_target = critic_target
        self.average_reward_log = []
        self.episode_reward_log = []
        self.episode_len_log = []
        self.noise_std_log = []
        self.test_max_score_log = []
        self.test_mean_score_log = []
        self.Q_log = []
        self.critic_loss_log = []
        self.start = False

    def set_start_signal(self):
        self.start = True

    def get_start_signal(self):
        return self.start

    def get_weights(self):
        return (self.actor.get_weights(), self.actor_target.get_weights(), self.critic.get_weights(), self.critic_target.get_weights())

    def set_weights(self, actor_weights, actor_target_weights, critic_weights, critic_target_weights):
        self.actor.set_weights(actor_weights)
        self.actor_target.set_weights(actor_target_weights)
        self.critic.set_weights(critic_weights)
        self.critic_target.set_weights(critic_target_weights)

    def incr_counter(self):
        self.step_counter += 1

    def get_counter(self):
        return self.step_counter

    def add_actor_log(self, average_reward, episode_reward, episode_len, noise_std):
        self.average_reward_log.append(average_reward)
        self.episode_reward_log.append(episode_reward)
        self.episode_len_log.append(episode_len)
        self.noise_std_log.append(noise_std)

    def add_test_log(self, test_max_score, test_mean_score):
        self.test_max_score_log.append(test_max_score)
        self.test_mean_score_log.append(test_mean_score)

    def add_learner_log(self, Q, critic_loss):
        self.Q_log.append(Q)
        self.critic_loss_log.append(critic_loss)

    def get_log(self):
        average_reward = None if len(self.average_reward_log) == 0 else self.average_reward_log.pop()
        episode_reward = None if len(self.episode_reward_log) == 0 else self.episode_reward_log.pop()
        episode_len = None if len(self.episode_len_log) == 0 else self.episode_len_log.pop()
        noise_std = None if len(self.noise_std_log) == 0 else self.noise_std_log.pop()
        test_max_score = None if len(self.test_max_score_log) == 0 else self.test_max_score_log.pop()
        test_mean_score = None if len(self.test_mean_score_log) == 0 else self.test_mean_score_log.pop()
        Q = None if len(self.Q_log) == 0 else self.Q_log.pop()
        critic_loss = None if len(self.critic_loss_log) == 0 else self.critic_loss_log.pop()

        return [average_reward, episode_reward, episode_len, noise_std, test_max_score, test_mean_score, Q, critic_loss]

class LocalBuffer(object):
    def __init__(self, parameters, device, settings):
        self.parameters = parameters
        self.state_dim = parameters['state_dim']
        self.num_actions = parameters['action_dim']
        self.batch_size = parameters['batch_size']
        self.max_size = int(parameters['batch_size'])
        self.buffer_size = int(parameters['batch_size'])
        self.device = device
        self.settings = settings

        # Prioritized Experience Replay
        self.alpha = 0.6
        self.priorities = np.ones((self.max_size, 1))
        self.beta = 0.4

        self.ptr = 0
        self.crt_size = 0

        self.state = np.zeros((self.max_size, self.state_dim))
        self.action = np.zeros((self.max_size, self.num_actions))
        self.action_high = np.zeros((self.max_size, self.num_actions))
        self.action_low = np.zeros((self.max_size, self.num_actions))
        self.next_state = np.array(self.state)
        self.next_action_high = np.zeros((self.max_size, self.num_actions))
        self.next_action_low = np.zeros((self.max_size, self.num_actions))
        self.reward = np.zeros((self.max_size, 1))
        self.done = np.zeros((self.max_size, 1))

    def add(self, state, action, action_high, action_low, next_state, next_action_high, next_action_low, reward, done):
        self.state[self.ptr] = state
        if self.parameters['only_power']:
            if self.parameters['only_thermal']:
                self.action[self.ptr] = np.asarray(action['adjust_gen_p'])[self.settings.thermal_ids]
            else:
                self.action[self.ptr] = action['adjust_gen_p']
        else:
            self.action[self.ptr] = np.asarray([action['adjust_gen_p'], action['adjust_gen_v']]).flatten()
        self.action_high[self.ptr] = action_high
        self.action_low[self.ptr] = action_low
        self.next_state[self.ptr] = next_state
        self.next_action_high[self.ptr] = next_action_high
        self.next_action_low[self.ptr] = next_action_low
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)

    def get_whole_buffer(self):
        return (
            torch.from_numpy(self.state).float().to(self.device),
            torch.from_numpy(self.action).float().to(self.device),
            torch.from_numpy(self.action_high).float().to(self.device),
            torch.from_numpy(self.action_low).float().to(self.device),
            torch.from_numpy(self.next_state).float().to(self.device),
            torch.from_numpy(self.next_action_high).float().to(self.device),
            torch.from_numpy(self.next_action_low).float().to(self.device),
            torch.from_numpy(self.reward).float().to(self.device),
            torch.from_numpy(self.done).float().to(self.device)
        )

    # def get_priorities(self):
    #     return self.priorities
    #
    # def update_priorities(self, priorities):
    #     for i in range(len(self.priorities)):
    #         self.priorities[i] = priorities[i]

@ray.remote(num_cpus=0.5, num_gpus=0.05)
class Actor_DDPG(object):
    def __init__(self, id, shared_memory, storage, parameters, settings, device):
        self.id = id
        self.shared_memory = shared_memory
        self.storage = storage
        self.parameters = parameters
        self.device = device
        self.settings = settings
        self.local_buffer = LocalBuffer(parameters, device, settings)

        self.gamma = parameters['gamma']

        self.agent = DDPG_Agent(settings, device, parameters['action_dim'], parameters['state_dim'], parameters)
        self.agent.actor.eval()
        self.agent.actor_target.eval()
        self.agent.critic.eval()
        self.agent.critic_target.eval()
        self.rand_agent = RandomAgent(settings.num_gen)

        self.env = Environment(settings, "EPRIReward")
        self.std_noise_schedule = LinearSchedule(parameters['training_iterations'], final_p=0.0, initial_p=0.3)
        self.epsilon_schedule = LinearSchedule(parameters['training_iterations'], final_p=0.0, initial_p=0.9)

        self.total_tranition = self.parameters['total_transitions'] // self.parameters['actor_num']
        self.last_model_index = -1

    def run(self):
        obs, done = self.env.reset(), False
        state = get_state_from_obs(obs, self.settings)
        action_high, action_low = get_action_space(obs, self.parameters, self.settings)
        time_step = 0
        episode_num = 0
        episode_reward = 0.
        episode_timesteps = 0
        average_reward = 0.
        noise_std = 0.
        log_count = 0
        start_training = False
        while True:
            start_training = ray.get(self.shared_memory.get_start_signal.remote())
            train_step = ray.get(self.shared_memory.get_counter.remote())
            episode_timesteps += 1
            if self.parameters['random_explore'] == 'EpsGreedy':
                # Epsilon-Greedy
                if np.random.uniform(0, 1) < self.epsilon_schedule.value(time_step):
                    action = self.rand_agent.act(obs)
                else:
                    action = self.agent.act(state, obs)
                    check_extreme_action(action, action_high, action_low, self.settings, self.parameters['only_power'],
                                         self.parameters['only_thermal'])
            elif self.parameters['random_explore'] == 'Gaussian':
                # Gaussian Noise
                action = self.agent.act(state, obs)
                check_extreme_action(action, action_high, action_low, self.settings, self.parameters['only_power'],
                                     self.parameters['only_thermal'])
                noise_std = self.std_noise_schedule.value(time_step)
                action = add_normal_noise(noise_std, action, action_high, action_low, self.settings,
                                          self.parameters['only_power'], self.parameters[
                                              'only_thermal'])  # add normal noise on action to improve exploration
            elif self.parameters['random_explore'] == 'none':
                action = self.agent.act(state, obs)
                check_extreme_action(action, action_high, action_low, self.settings, self.parameters['only_power'],
                                     self.parameters['only_thermal'])

            next_obs, reward, done, info = self.env.step(action)
            episode_reward += reward
            if sum(np.asarray(next_obs.gen_p)[self.settings.renewable_ids]) > sum(next_obs.load_p):
                print('renewable power is more ...')
            next_state = get_state_from_obs(next_obs, self.settings)
            next_action_high, next_action_low = get_action_space(next_obs, self.parameters, self.settings)

            self.local_buffer.add(state, action, action_high, action_low, next_state, next_action_high, next_action_low, reward, np.float(done))

            if time_step % self.parameters['batch_size'] == 0 and time_step > 0:
                state_b, action_b, action_high_b, action_low_b, next_state_b, next_action_high_b, next_action_low_b, reward_b, done_b = self.local_buffer.get_whole_buffer()

                # Compute the target Q value using the information of next state
                next_action_b = self.agent.actor_target(next_state_b, next_action_high_b, next_action_low_b)
                Q_tmp = self.agent.critic_target(next_state_b, next_action_b)
                Q_target = reward_b + self.gamma * (1 - done_b) * Q_tmp
                Q_current = self.agent.critic(state_b, action_b)

                priorities = L1Loss(reduction='none')(Q_current, Q_target).data.cpu().numpy() + 1e-6
                self.storage.push_trajectory((state_b.detach().cpu().numpy(), action_b.detach().cpu().numpy(),
                                        action_high_b.detach().cpu().numpy(), action_low_b.detach().cpu().numpy(),
                                        next_state_b.detach().cpu().numpy(), next_action_high_b.detach().cpu().numpy(),
                                        next_action_low_b.detach().cpu().numpy(), reward_b.detach().cpu().numpy(),
                                        done_b.detach().cpu().numpy(), priorities))

            new_model_index = train_step // self.parameters['actor_update_interval']
            if new_model_index > self.last_model_index:
                print('=======actor model updated========')
                self.last_model_index = new_model_index
                actor_weights, actor_target_weights, critic_weights, critic_target_weights = ray.get(self.shared_memory.get_weights.remote())
                self.agent.actor.set_weights(actor_weights)
                self.agent.actor.to(self.device)
                self.agent.actor.eval()
                self.agent.actor_target.set_weights(actor_target_weights)
                self.agent.actor_target.to(self.device)
                self.agent.actor_target.eval()
                self.agent.critic.set_weights(critic_weights)
                self.agent.critic.to(self.device)
                self.agent.critic.eval()
                self.agent.critic_target.set_weights(critic_target_weights)
                self.agent.critic_target.to(self.device)
                self.agent.critic_target.eval()

            if done:
                # Reset environment
                obs, done = self.env.reset(), False
                state = get_state_from_obs(obs, self.settings)
                action_high, action_low = get_action_space(obs, self.parameters, self.settings)
                average_reward = episode_reward / episode_timesteps

                episode_num += 1

                # train_step = ray.get(self.shared_memory.get_counter.remote())
                if self.id == 0 and train_step > log_count*self.parameters['log_interval'] - 100 and train_step < (log_count+8)*self.parameters['log_interval']:
                    self.shared_memory.add_actor_log.remote(average_reward, episode_reward, episode_timesteps, noise_std)
                    log_count += 1

                episode_reward = 0
                episode_timesteps = 0

            elif start_training and (time_step / self.total_tranition) > (train_step / self.parameters['training_iterations']):
                print('-------------actor wait for a moment-------------')
                time.sleep(1)
                continue

            time_step += 1


class DDPG_Agent(BaseAgent):
    def __init__(
            self,
            settings,
            device,
            action_dim,
            state_dim,
            parameters
        ):

        BaseAgent.__init__(self, settings.num_gen)

        self.device = device
        self.settings = settings
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.state_shape = (-1, state_dim)

        self.parameters = parameters
        self.gamma = parameters['gamma']
        self.tau = parameters['tau']

        self.actor = ActorNet(state_dim, action_dim, self.settings).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = CriticNet(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

    def act(self, state, obs, done=False):
        self.actor.eval()
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        action_high, action_low = get_action_space(obs, self.parameters, self.settings)
        action_high, action_low = torch.from_numpy(action_high).unsqueeze(0).to(self.device), torch.from_numpy(action_low).unsqueeze(0).to(self.device)
        adjust_gen = self.actor(state, action_high, action_low).squeeze().detach().cpu().numpy()

        if self.parameters['only_power']:
            if self.parameters['only_thermal']:
                adjust_gen_renewable = adjust_renewable_generator(obs, self.settings)
                adjust_gen_p = form_p_action(adjust_gen, adjust_gen_renewable, self.settings)
            else:
                adjust_gen_p = adjust_gen
            adjust_gen_v = voltage_action(obs, self.settings)
        else:
            adjust_gen_p, adjust_gen_v = adjust_gen[:len(adjust_gen)//2], adjust_gen[len(adjust_gen)//2:]
        self.actor.train()
        return form_action(adjust_gen_p, adjust_gen_v)

@ray.remote(num_gpus=0.05)
class Testor(object):
    def __init__(self, settings, device, parameters, shared_memory):
        self.agent = DDPG_Agent(settings, device, parameters['action_dim'], parameters['state_dim'], parameters)
        self.agent.actor.eval()
        self.agent.actor_target.eval()
        self.agent.critic.eval()
        self.agent.critic_target.eval()
        self.shared_memory = shared_memory
        self.parameters = parameters
        self.device = device
        self.settings = settings

        self.test_count = 0

    def run_task(self, actor_weights):
        self.agent.actor.set_weights(actor_weights)
        self.agent.actor.to(self.device)
        self.agent.actor.eval()
        max_episode = 10
        episode_reward = [0 for _ in range(max_episode)]
        for episode in range(max_episode):
            print('------ episode ', episode)
            env = Environment(settings, "EPRIReward")
            print('------ reset ')
            obs = env.reset()

            done = False
            timestep = 0
            while not done:
                print('------ step ', timestep)
                state = get_state_from_obs(obs, settings)
                action = self.agent.act(state, obs)
                obs, reward, done, info = env.step(action)
                episode_reward[episode] += reward
                timestep += 1
                if done:
                    obs = env.reset()
                    print('info:', info)
                    print(f'episode cumulative reward={episode_reward[episode]}')

        return max(episode_reward), sum(episode_reward) / len(episode_reward)

    def run(self):
        while True:
            step_count = ray.get(self.shared_memory.get_counter.remote())
            if step_count > self.test_count*self.parameters['test_interval'] - 100 and step_count < (self.test_count+2)*self.parameters['test_interval']:
                actor_weights, _, _, _ = ray.get(self.shared_memory.get_weights.remote())
                test_max_score, test_mean_score = self.run_task(actor_weights)
                self.shared_memory.add_test_log.remote(test_max_score, test_mean_score)
                self.test_count += 1
            else:
                time.sleep(10)