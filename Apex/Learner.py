import numpy as np
from ReplayBuffer import LinearSchedule
import torch
import time
import ray
from DDPG import ActorNet, CriticNet
from torch.nn import L1Loss
import copy

@ray.remote
class SharedBuffer(object):
    def __init__(self, parameters, device, settings, storage):
        self.parameters = parameters
        self.state_dim = parameters['state_dim']
        self.num_actions = parameters['action_dim']
        self.batch_size = parameters['batch_size']
        self.max_size = int(parameters['buffer_size'])
        self.buffer_size = int(parameters['buffer_size'])
        self.device = device
        self.settings = settings

        self.storage = storage

        # Prioritized Experience Replay
        self.alpha = 0.6
        self.priorities = np.ones((self.max_size, 1))
        self.beta = 0.4
        # self.beta_schedule = LinearSchedule(parameters['max_timestep'], final_p=1.0, initial_p=self.init_beta)

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

    def add_trajectory(self, state, action, action_high, action_low, next_state, next_action_high, next_action_low, reward, done, priorities):
        self.state[self.ptr:self.ptr+self.batch_size] = state
        if self.parameters['only_power']:
            if self.parameters['only_thermal']:
                self.action[self.ptr:self.ptr+self.batch_size] = action
            else:
                self.action[self.ptr:self.ptr+self.batch_size] = action['adjust_gen_p']
        else:
            self.action[self.ptr:self.ptr+self.batch_size] = np.asarray([action['adjust_gen_p'], action['adjust_gen_v']]).flatten()
        self.action_high[self.ptr:self.ptr+self.batch_size] = action_high
        self.action_low[self.ptr:self.ptr+self.batch_size] = action_low
        self.next_state[self.ptr:self.ptr+self.batch_size] = next_state
        self.next_action_high[self.ptr:self.ptr+self.batch_size] = next_action_high
        self.next_action_low[self.ptr:self.ptr+self.batch_size] = next_action_low
        self.reward[self.ptr:self.ptr+self.batch_size] = reward
        self.done[self.ptr:self.ptr+self.batch_size] = done
        self.priorities[self.ptr:self.ptr+self.batch_size] = priorities

        self.ptr = (self.ptr + self.batch_size) % self.max_size
        print(f'ptr={self.ptr}')
        self.crt_size = min(self.crt_size + self.batch_size, self.max_size)

    def sample_batch(self):
        probs = self.priorities[:self.crt_size] ** self.alpha
        probs /= probs.sum()
        probs = np.squeeze(probs)
        ind = np.random.choice(self.crt_size, self.batch_size, p=probs, replace=False)

        weights_lst = (self.crt_size * probs[ind]) ** (-self.beta)
        weights_lst /= weights_lst.max()

        return (
            self.state[ind], self.action[ind], self.action_high[ind], self.action_low[ind],
            self.next_state[ind], self.next_action_high[ind], self.next_action_low[ind],
            self.reward[ind], self.done[ind],
            ind, weights_lst
        )

    def update_priorities(self, batch_indices, batch_priorities):
        for i in range(len(batch_indices)):
            idx, prior = batch_indices[i], batch_priorities[i]
            self.priorities[idx] = prior

    def run(self):
        while True:
            trajectory = self.storage.pop_trajectory()
            if trajectory is not None:
                state, action, action_high, action_low, next_state, next_action_high, next_action_low, reward, done, origin_priorities = trajectory
                self.add_trajectory(state, action, action_high, action_low, next_state, next_action_high,
                                    next_action_low, reward, done, origin_priorities)

            priorities = self.storage.pop_priority()
            if priorities is not None:
                ind, priors = priorities
                self.update_priorities(ind, priors)

            if self.ptr >= self.batch_size:
                self.storage.push_batch(self.sample_batch())
                # print('push 1 batch')

# @ray.remote(num_gpus=0.3)
class Learner_DDPG(object):
    def __init__(self, id, shared_memory, storage, parameters, settings, device, summary_writer):
        self.id = id
        self.shared_memory = shared_memory
        self.storage = storage
        self.parameters = parameters
        self.device = device
        self.settings = settings
        self.summary_writer = summary_writer

        self.time_step = 0

        self.gamma = parameters['gamma']
        self.tau = parameters['tau']

        self.actor = ActorNet(self.parameters['state_dim'], self.parameters['action_dim'], self.settings).to(self.device)
        self.actor.train()
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.eval()
        self.critic = CriticNet(self.parameters['state_dim'], self.parameters['action_dim']).to(self.device)
        self.critic.train()
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=parameters['optimizer_parameters']['lr'],
                                                weight_decay=parameters['optimizer_parameters']['weight_decay'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=parameters['optimizer_parameters']['lr'] / 10,
                                                 weight_decay=parameters['optimizer_parameters']['weight_decay'])

    def _train(self, batch):
        state, action, action_high, action_low, next_state, next_action_high, next_action_low, reward, done, ind, weights_lst = batch
        state = torch.from_numpy(state).float().to(self.device)
        action = torch.from_numpy(action).float().float().to(self.device)
        action_high = torch.from_numpy(action_high).float().to(self.device)
        action_low = torch.from_numpy(action_low).float().to(self.device)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        next_action_high = torch.from_numpy(next_action_high).float().to(self.device)
        next_action_low = torch.from_numpy(next_action_low).float().to(self.device)
        reward = torch.from_numpy(reward).float().to(self.device)
        done = torch.from_numpy(done).float().to(self.device)
        weights = torch.from_numpy(weights_lst).float().to(self.device).float()

        # Compute the target Q value using the information of next state
        next_action = self.actor_target(next_state, next_action_high, next_action_low)
        Q_tmp = self.critic_target(next_state, next_action)
        Q_target = reward + self.gamma * (1 - done) * Q_tmp
        Q_current = self.critic(state, action)

        # td_errors = L1Loss(reduction='none')(Q, Q_target)
        td_errors = Q_target - Q_current
        priorities = L1Loss(reduction='none')(Q_current, Q_target).data.cpu().numpy() + 1e-6
        self.storage.push_priority((ind, priorities))

        # Compute the current Q value and the loss
        critic_loss = torch.mean(weights * (td_errors ** 2))  # with importance sampling
        # critic_loss = nn.MSELoss()(Q_current, Q_target)     # without importance sampling

        # Optimize the critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2)
        self.critic_optimizer.step()

        # Make action and evaluate its action values
        action_out = self.actor(state, action_high, action_low)
        Q = self.critic(state, action_out)
        actor_loss = -torch.mean(Q)

        # Optimize the actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2)
        self.actor_optimizer.step()

        # if self.time_step % 10 == 0:
        #     print(f'actor gradient max={max([np.abs(p).max() for p in self.actor.get_gradients()])}')
        #     print(f'critic gradient max={max([np.abs(p).max() for p in self.critic.get_gradients()])}')
        #     print(f'actor loss={actor_loss:.3f}, critic loss={critic_loss:.3f}')

        return {
            'training/Q': Q_current.mean().detach().cpu().numpy(),
            'training/critic_loss': critic_loss.mean().detach().cpu().numpy(),
        }

    def copy_target_update(self):

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def _log(self, log_info):
        average_reward, episode_reward, episode_len, noise_std, test_max_score, test_mean_score, Q, critic_loss = log_info
        if average_reward is not None:
            self.summary_writer.add_scalar('episode/average_reward', average_reward, self.time_step)
            self.summary_writer.add_scalar('episode/cumulative_reward', episode_reward, self.time_step)
            self.summary_writer.add_scalar('episode/total_steps', episode_len, self.time_step)
            self.summary_writer.add_scalar('statistics/std', noise_std, self.time_step)
        if test_mean_score is not None:
            self.summary_writer.add_scalar('test/episodic_mean_score', test_mean_score, self.time_step)
        if Q is not None:
            self.summary_writer.add_scalar('training/Q', Q, self.time_step)
            self.summary_writer.add_scalar('training/critic_loss', critic_loss, self.time_step)

    def run(self):
        start_training = False
        while self.time_step < self.parameters['training_iterations']:
            batch = self.storage.pop_batch()
            if batch is None:
                # print('no batch...')
                time.sleep(0.2)
                continue

            if not start_training:
                self.shared_memory.set_start_signal.remote()
                start_training = True
                print('====-----training begin-----=====')

            train_info = self._train(batch)
            self.time_step += 1
            self.shared_memory.incr_counter.remote()

            if self.time_step % self.parameters['target_update_interval'] == 0:
                self.copy_target_update()

            if self.time_step % self.parameters['actor_update_interval'] == 0:
                self.shared_memory.set_weights.remote(self.actor.get_weights(), self.actor_target.get_weights(),
                                                      self.critic.get_weights(), self.critic_target.get_weights())

            if self.time_step % self.parameters['log_interval'] == 0:
                self.shared_memory.add_learner_log.remote(train_info['training/Q'], train_info['training/critic_loss'])
                log_info = ray.get(self.shared_memory.get_log.remote())
                self._log(log_info)


