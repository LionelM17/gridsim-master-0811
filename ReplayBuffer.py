import numpy as np
import torch

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

class StandardBuffer(object):
    def __init__(self, state_dim, num_actions, parameters, device, settings):
        self.parameters = parameters
        self.batch_size = parameters['batch_size']
        self.max_size = int(parameters['buffer_size'])
        self.buffer_size = int(parameters['buffer_size'])
        self.device = device
        self.settings = settings

        # Prioritized Experience Replay
        self.alpha = 0.6
        self.priorities = np.ones((self.max_size, 1))
        self.init_beta = 0.4
        self.beta_schedule = LinearSchedule(parameters['max_timestep'], final_p=1.0, initial_p=self.init_beta)

        self.ptr = 0
        self.crt_size = 0

        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, num_actions))
        self.action_high = np.zeros((self.max_size, num_actions))
        self.action_low = np.zeros((self.max_size, num_actions))
        self.next_state = np.array(self.state)
        self.next_action_high = np.zeros((self.max_size, num_actions))
        self.next_action_low = np.zeros((self.max_size, num_actions))
        self.reward = np.zeros((self.max_size, 1))
        self.done = np.zeros((self.max_size, 1))

    def add(self, state, action, action_high, action_low, next_state, next_action_high, next_action_low, reward, done, episode_start):
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

    def sample(self, time_step):
        # ind = np.random.randint(
        #     max(0, self.crt_size - self.buffer_size), self.crt_size, size=self.batch_size)

        probs = self.priorities[:self.crt_size] ** self.alpha
        probs /= probs.sum()
        probs = np.squeeze(probs)
        ind = np.random.choice(self.crt_size, self.batch_size, p=probs, replace=False)

        # probs_arg = np.argsort(probs[ind])
        # reward_arg = np.argsort(self.reward[ind])
        # import ipdb
        # ipdb.set_trace()

        beta = self.beta_schedule.value(time_step)
        weights_lst = (self.crt_size * probs[ind]) ** (-beta)
        weights_lst /= weights_lst.max()

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.action_high[ind]).to(self.device),
            torch.FloatTensor(self.action_low[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.next_action_high[ind]).to(self.device),
            torch.FloatTensor(self.next_action_low[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device),
            ind,
            weights_lst
        )

    def update_priorities(self, batch_indices, batch_priorities):
        for i in range(len(batch_indices)):
            idx, prior = batch_indices[i], batch_priorities[i]
            self.priorities[idx] = prior
