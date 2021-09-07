from Agent.BaseAgent import BaseAgent
from DDPG import ActorNet, CriticNet
from utils import get_action_space, voltage_action, adjust_renewable_generator, form_p_action
from utilize.form_action import *
import torch.nn as nn
from torch.nn import L1Loss
import numpy as np

import torch
import copy

class DDPG_Agent(BaseAgent):
    def __init__(
            self,
            settings,
            replay_buffer,
            device,
            action_dim,
            state_dim,
            parameters
        ):

        BaseAgent.__init__(self, settings.num_gen)

        self.device = device
        self.replay_buffer = replay_buffer
        self.settings = settings
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.state_shape = (-1, state_dim)

        self.parameters = parameters
        self.gamma = parameters['gamma']
        self.tau = parameters['tau']
        self.initial_eps = parameters['initial_eps']
        self.end_eps = parameters['end_eps']
        self.eps_decay = parameters['eps_decay']

        self.actor = ActorNet(state_dim, action_dim, self.settings).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = CriticNet(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=parameters['optimizer_parameters']['lr'], weight_decay=parameters['optimizer_parameters']['weight_decay'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=parameters['optimizer_parameters']['lr'] / 10, weight_decay=parameters['optimizer_parameters']['weight_decay'])
        self.cnt = 0

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
        # self.actor.train()
        return form_action(adjust_gen_p, adjust_gen_v)

    def copy_target_update(self):
        # Softly update the target networks
        # actor_dict = self.actor_target.state_dict()
        # critic_dict = self.critic_target.state_dict()
        # for x in self.actor_target.state_dict().keys():
        #     import ipdb
        #     ipdb.set_trace()
        #     actor_dict[x] *= 1 - self.tau
        #     actor_dict[x] += self.tau * self.actor.state_dict()[x]
        # for x in self.critic_target.state_dict().keys():
        #     critic_dict[x] *= 1 - self.tau
        #     critic_dict[x] += self.tau * self.critic.state_dict()[x]
        # self.actor_target.load_state_dict(actor_dict)
        # self.critic_target.load_state_dict(critic_dict)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def train(self, time_step):
        self.actor.train()
        self.critic.train()

        for epoch in range(self.parameters['K_epochs']):
            # Sample replay buffer
            state, action, action_high, action_low, next_state, next_action_high, next_action_low, reward, done, ind, weights_lst = self.replay_buffer.sample(time_step)
            weights = torch.from_numpy(weights_lst).to(self.device).float()

            # Compute the target Q value using the information of next state
            # next_state = next_state.to(self.device)
            next_action = self.actor_target(next_state, next_action_high, next_action_low)
            Q_tmp = self.critic_target(next_state, next_action)
            Q_target = reward + self.gamma * (1 - done) * Q_tmp
            Q_current = self.critic(state, action)

            # td_errors = L1Loss(reduction='none')(Q, Q_target)
            td_errors = Q_target - Q_current
            priorities = L1Loss(reduction='none')(Q_current, Q_target).data.cpu().numpy() + 1e-6
            self.replay_buffer.update_priorities(ind, priorities)

            # Compute the current Q value and the loss
            critic_loss = torch.mean(weights * (td_errors ** 2))   # with importance sampling
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

            if self.cnt % 10 == 0:
                print(f'actor gradient max={max([np.abs(p).max() for p in self.actor.get_gradients()])}')
                print(f'critic gradient max={max([np.abs(p).max() for p in self.critic.get_gradients()])}')
                print(f'actor loss={actor_loss:.3f}, critic loss={critic_loss:.3f}')
            self.cnt += 1

        return {
            'training/Q': Q_current.mean().detach().cpu().numpy(),
            'training/critic_loss': critic_loss.mean().detach().cpu().numpy(),
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