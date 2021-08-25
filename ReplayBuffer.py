import numpy as np
import torch

class StandardBuffer(object):
    def __init__(self, state_dim, num_actions, parameters, device):
        self.parameters = parameters
        self.batch_size = parameters['batch_size']
        self.max_size = int(parameters['buffer_size'])
        self.buffer_size = int(parameters['buffer_size'])
        self.device = device

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

    def sample(self):
        ind = np.random.randint(
            max(0, self.crt_size - self.buffer_size), self.crt_size, size=self.batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.action_high[ind]).to(self.device),
            torch.FloatTensor(self.action_low[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.next_action_high[ind]).to(self.device),
            torch.FloatTensor(self.next_action_low[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )