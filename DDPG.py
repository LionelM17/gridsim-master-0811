import gym
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Agent.BaseAgent import BaseAgent

class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNet, self).__init__()

        self.fc1 = nn.Linear(state_dim, 1024)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(1024, 256)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(256, action_dim)
        self.out.weight.data.normal_(0, 0.1)

        # self.l1 = nn.Linear(state_dim, 64)
        # self.l2 = nn.Linear(64, action_dim)

    def forward(self, state):
        q = F.relu(self.fc1(state))
        q = F.relu(self.fc2(q))
        q = self.out(q)
        # #q = torch.tanh(q)
        # q = torch.max(q)
        # action = q * 0.005 + 0.059
        return q


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

class DDPG_Agent(BaseAgent):
    def __init__(
            self,
            num_gen,
            action_dim,
            state_dim,
            gamma = 0.99,
            optimizer = "Adam",
            optimizer_paoameters = {},
            tau = 0.001,
            initial_eps = 1.0,
            end_eps = 0.001,
            eps_decay = 5e3,
        ):

        BaseAgent.__init__(num_gen)

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.state_shape = (-1, state_dim)

        self.gamma = gamma
        self.tau = tau
        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.eps_decay = eps_decay

        self.actor = ActorNet(state_dim, action_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = CriticNet(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.tau)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.tau)

    def act(self, obs, reward, done=False):
        s = torch.unsqueeze(torch.FloatTensor(obs), 0)
        action = self.actor(s)[0].detach()
        action = action / torch.max(action)
        action = action * 0.6 + 0.6
        return action

    def copy_target_update(self):
        # Softly update the target networks
        for x in self.actor_target.state_dict().keys():
            eval('self.actor_target.' + x + '.data.mul_((1-self.tau))')
            eval('self.actor_target.' + x + '.data.add_(self.tau*self.actor.' + x + '.data)')
        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_((1-self.tau))')
            eval('self.critic_target.' + x + '.data.add_(self.tau*self.critic.' + x + '.data)')

    def train(self, replay_buffer):
        # Sample replay buffer
        state, action, next_state, reward, done = replay_buffer.sample()

        # Make action and evaluate its action values
        action_out = self.actor(state)
        Q = self.critic(state, action_out)
        actor_loss = -torch.mean(Q)

        # Optimize the actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Compute the target Q value using the information of next state
        action_target = self.actor_target(next_state)
        Q_tmp = self.critic_target(next_state, action_target)
        Q_target = reward + self.gamma * Q_tmp

        # Compute the current Q value and the loss
        Q_current = self.critic(state, action)
        critic_loss = nn.MSELoss()(Q_target, Q_current)

        # Optimize the critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

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
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

    def add(self, state, action, next_state, reward, done, episode_done, episode_start):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
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
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
