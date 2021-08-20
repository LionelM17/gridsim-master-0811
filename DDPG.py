import gym
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
from math import *

from Agent.BaseAgent import BaseAgent
from utilize.form_action import *

class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, settings):
        super(ActorNet, self).__init__()
        self.settings = settings

        self.fc1 = nn.Linear(state_dim, 1024)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(1024, 256)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(256, action_dim)
        self.out.weight.data.normal_(0, 0.1)
        # TODO: add scale layer
        # self.l1 = nn.Linear(state_dim, 64)
        # self.l2 = nn.Linear(64, action_dim)

    def forward(self, state, action_high, action_low):
        m = self.fc1(state)
        m = self.fc2(m)
        m = self.out(m)
        m = torch.sigmoid(m)
        m = (m + torch.ones_like(m).to('cuda')) / (2*torch.ones_like(m).to('cuda'))\
            * (action_high - action_low) + action_low
        return m

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads


class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNet, self).__init__()
        self.fcs = nn.Linear(state_dim, 1024)
        self.fcs.weight.data.normal_(0, 0.1)
        self.fca = nn.Linear(action_dim, 1024)
        self.fca.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(1024, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, state, action):
        x = self.fcs(state)
        y = self.fca(action)
        action_value = self.out(F.relu(x + y))
        return action_value

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

def get_action_space(obs, settings):
    action_high = np.asarray([obs.action_space['adjust_gen_p'].high, obs.action_space['adjust_gen_v'].high]).flatten()
    action_high = np.where(np.isinf(action_high), np.full_like(action_high, 0), action_high)
    action_low = np.asarray([obs.action_space['adjust_gen_p'].low, obs.action_space['adjust_gen_v'].low]).flatten()
    action_low = np.where(np.isinf(action_low), np.full_like(action_low, 0), action_low)
    action_high = torch.from_numpy(action_high)
    action_low = torch.from_numpy(action_low)
    return action_high, action_low

# ???????
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

class DDPG_Agent(BaseAgent):
    def __init__(
            self,
            settings,
            device,
            action_dim,
            state_dim,
            gamma=0.99,
            tau=0.001,
            initial_eps=1.0,
            end_eps=0.001,
            eps_decay=0.999,
        ):

        BaseAgent.__init__(self, settings.num_gen)

        self.device = device

        self.settings = settings
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.state_shape = (-1, state_dim)

        self.gamma = gamma
        self.tau = tau
        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.eps_decay = eps_decay

        self.actor = ActorNet(state_dim, action_dim, self.settings).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = CriticNet(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.tau)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.tau)
        self.cnt = 0

    def act(self, state, obs, done=False):
        state = state.to(self.device)
        action_high, action_low = get_action_space(obs, self.settings)
        action_high, action_low = action_high.to(self.device), action_low.to(self.device)
        adjust_gen = self.actor_target(state, action_high, action_low).detach().cpu().numpy()
        adjust_gen_p, adjust_gen_v = adjust_gen[:len(adjust_gen)//2], adjust_gen[len(adjust_gen)//2:]
        return form_action(adjust_gen_p, adjust_gen_v)

    def copy_target_update(self):
        # Softly update the target networks
        for x in self.actor_target.state_dict().keys():
            eval('self.actor_target.' + x + '.data.mul_((1-self.tau))')
            eval('self.actor_target.' + x + '.data.add_(self.tau*self.actor.' + x + '.data)')
        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_((1-self.tau))')
            eval('self.critic_target.' + x + '.data.add_(self.tau*self.critic.' + x + '.data)')

    def train(self, replay_buffer, obs, next_obs):
        # Sample replay buffer
        state, action, next_state, reward, done = replay_buffer.sample()
        state = state.to(self.device)

        # Make action and evaluate its action values
        # TODO: is legalize_action necessary in train ?
        action_high, action_low = get_action_space(obs, self.settings)
        action_high, action_low = action_high.to(self.device), action_low.to(self.device)
        action_out = self.actor(state, action_high, action_low)

        Q = self.critic(state, action_out)
        actor_loss = -torch.mean(Q)

        # Optimize the actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Compute the target Q value using the information of next state
        next_state = next_state.to(self.device)
        action_high, action_low = get_action_space(next_obs, self.settings)
        action_high, action_low = action_high.to(self.device), action_low.to(self.device)
        action_target = self.actor_target(next_state, action_high, action_low)
        Q_tmp = self.critic_target(next_state, action_target)
        Q_target = reward + self.gamma * Q_tmp

        # Compute the current Q value and the loss
        Q_current = self.critic(state, action)
        critic_loss = nn.MSELoss()(Q_target, Q_current)

        # Optimize the critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.cnt % 50 == 0:
            print(f'actor gradient max={max([p.max() for p in self.actor.get_gradients()])}')
            print(f'critic gradient max={max([p.max() for p in self.critic.get_gradients()])}')
        self.critic_optimizer.step()
        self.cnt += 1
        return {
            'Q': Q_current.mean().detach().cpu().numpy(),
            'Q_loss': critic_loss.mean().detach().cpu().numpy(),
        }

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_DDPG_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "DDPG_actor_optimizer")
        torch.save(self.critic.state_dict(), filename + "_DDPG_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_DDPG_critic_optimizer")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_DDPG_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_DDPG_actor_optimizer"))
        self.critic.load_state_dict(torch.load(filename + "_DDPG_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_DDPG_critic_optimizer"))

class StandardBuffer(object):
    def __init__(self, state_dim, num_actions, batch_size, buffer_size, device):
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.buffer_size = int(buffer_size)
        self.device = device

        self.ptr = 0
        self.crt_size = 0

        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, num_actions))
        # self.action_ori = np.zeros((self.max_size, num_actions))
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

    def add(self, state, action, next_state, reward, done, episode_start):
        self.state[self.ptr] = state
        self.action[self.ptr] = np.asarray([action['adjust_gen_p'], action['adjust_gen_v']]).flatten()
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)

    def sample(self):
        ind = np.random.randint(
            max(0, self.crt_size - self.buffer_size), self.crt_size, size=self.batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            # torch.FloatTensor(self.action_ori[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
