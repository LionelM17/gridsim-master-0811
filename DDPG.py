import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, settings):
        super(ActorNet, self).__init__()
        self.settings = settings

        # self.fc1 = nn.Linear(state_dim, 1024)
        # # self.fc1.weight.data.normal_(0, 0.1)
        # self.bn1 = nn.BatchNorm1d(1024, momentum=0.1)
        # self.fc2 = nn.Linear(1024, 256)
        # # self.fc2.weight.data.normal_(0, 0.1)
        # self.bn2 = nn.BatchNorm1d(256, momentum=0.1)
        # self.out = nn.Linear(256, action_dim)
        # # self.out.weight.data.normal_(0, 0.1)
        # # TODO: add scale layer
        # # self.l1 = nn.Linear(state_dim, 64)
        # # self.l2 = nn.Linear(64, action_dim)

        self.fc = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
            nn.Sigmoid()
        )

    def forward(self, state, action_high, action_low):
        # m = self.fc1(state)
        # m = self.bn1(m)
        # m = self.fc2(m)
        # m = self.bn2(m)
        # m = self.out(m)
        # m = torch.tanh(m)
        m = self.fc(state)
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
        # self.fcs = nn.Linear(state_dim, 1024)
        # # self.fcs.weight.data.normal_(0, 0.1)
        # # self.bn1 = nn.BatchNorm1d(1024, momentum=0.1)
        # self.fca = nn.Linear(action_dim, 1024)
        # # self.fca.weight.data.normal_(0, 0.1)
        # # self.bn2 = nn.BatchNorm1d(1024, momentum=0.1)
        # self.out = nn.Linear(1024, 1)
        # # self.out.weight.data.normal_(0, 0.1)

        self.fcs = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 256)
        )
        self.fca = nn.Sequential(
            nn.Linear(action_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256)
        )
        self.out = nn.Sequential(
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = self.fcs(state)
        y = self.fca(action)
        action_value = self.out(x + y)
        return action_value

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

