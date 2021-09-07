import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y

nn.LayerNorm = LayerNorm

class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, settings):
        super(ActorNet, self).__init__()
        self.settings = settings

        self.fc = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.LayerNorm(1024),
            # nn.BatchNorm1d(num_features=1024, affine=False),
            # nn.Tanh(),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            # nn.BatchNorm1d(num_features=256, affine=False),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(256, action_dim),
            # nn.Sigmoid()
            nn.Tanh()
            # nn.Softmax()
        )

    def forward(self, state, action_high, action_low):
        m = self.fc(state)
        m = (m + torch.ones_like(m).to('cuda')) / (2*torch.ones_like(m).to('cuda'))\
            * (action_high - action_low) + action_low  # for Tanh or Sigmoid
        m = (m - (action_high / 2 + action_low / 2)) * 0.8 + (
                    action_high / 2 + action_low / 2)  # compressed action space
        # m = m * (action_high - action_low) + action_low  # for Softmax
        return m

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g).to('cuda')

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNet, self).__init__()

        self.fcs = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.LayerNorm(1024),
            # nn.BatchNorm1d(num_features=1024),
            nn.ReLU()
        )
        self.fca = nn.Sequential(
            nn.Linear(action_dim + 1024, 512),
            nn.LayerNorm(512),
            # nn.BatchNorm1d(num_features=512),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(512, 1)
        )

    def forward(self, state, action):
        # x = self.fcs(state)
        # y = self.fca(action)
        # action_value = self.out(x + y)
        # return action_value

        x = self.fcs(state)
        x = self.fca(torch.cat([x, action], 1))
        action_value = self.out(x)
        return action_value

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g).to('cuda')

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

