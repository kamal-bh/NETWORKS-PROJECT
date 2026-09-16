
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class Actor(nn.Module):
    def __init__(self, state_dim=2, action_dim=2, hidden=64):
        super(Actor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden),   # input layer: 2 → 64
            nn.ReLU(),                       # activation: adds non-linearity
            nn.Linear(hidden, hidden),       # hidden layer: 64 → 64
            nn.ReLU(),
            nn.Linear(hidden, action_dim),  # output layer: 64 → 2
            nn.Softmax(dim=-1),             # converts raw numbers to weights summing to 1
        )

    def forward(self, state):
        # state is a tensor; .forward() is called automatically when you do actor(state)
        return self.network(state)

class Critic(nn.Module):
    def __init__(self, state_dim=2, action_dim=2, hidden=64):
        super(Critic, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),  # input: 4 → 64
            nn.ReLU(),
            nn.Linear(hidden, hidden),                  # 64 → 64
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)  # [q1, q2, w1, w2]
        return self.network(x)


class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store one experience tuple."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        """Randomly pull out a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),       # shape: (64, 2)
            torch.FloatTensor(np.array(actions)),      # shape: (64, 2)
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),     # shape: (64, 1)
            torch.FloatTensor(np.array(next_states)), # shape: (64, 2)
            torch.FloatTensor(np.array(dones)).unsqueeze(1),       # shape: (64, 1)
        )

    def __len__(self):
        return len(self.buffer)


class OUNoise:
    def __init__(self, action_dim=2, mu=0.0, theta=0.15, sigma=0.2):
        """
        mu:    where the noise drifts toward (0 = no drift)
        theta: how strongly it pulls back toward mu (like a spring)
        sigma: how much randomness (width of noise)
        """
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset noise state to zero at the start of each episode."""
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        """Generate one noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state = self.state + dx
        return self.state


class DDPGAgent:
    def __init__(
        self,
        state_dim=2,
        action_dim=2,
        hidden=64,
        lr_actor=1e-3,   
        lr_critic=1e-3, 
        gamma=0.99,    
        tau=0.005,          # soft update rate for target networks
        buffer_capacity=10_000,
        batch_size=64,
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # ── Networks ──────────────────────────────────────────
        self.actor  = Actor(state_dim, action_dim, hidden)
        self.critic = Critic(state_dim, action_dim, hidden)
        self.actor_target  = Actor(state_dim, action_dim, hidden)
        self.critic_target = Critic(state_dim, action_dim, hidden)
        self._hard_copy(self.actor,  self.actor_target)   
        self._hard_copy(self.critic, self.critic_target)
        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = ReplayBuffer(buffer_capacity)
        self.noise  = OUNoise(action_dim)

    def select_action(self, state, add_noise=True, noise_scale=1.0):
        """
        Convert state to a tensor, pass through Actor, optionally add noise.

        add_noise=True during training (explore), False during evaluation (exploit).
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)  # shape: (1, 2)

        self.actor.eval()  # turn off dropout/batchnorm effects (we don't use them, but good practice)
        with torch.no_grad():  # don't track gradients (saves memory)
            action = self.actor(state_t).squeeze(0).numpy()  # shape: (2,)
        self.actor.train()

        if add_noise:
            noise = self.noise.sample() * noise_scale
            action = action + noise
            # Clip and re-normalize so weights stay valid
            action = np.clip(action, 0.01, None)
            action = action / action.sum()

        return action

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self):
        # Don't learn until we have enough memories
        if len(self.buffer) < self.batch_size:
            return None, None  # not ready yet

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)           # what actor would do next
            target_q     = self.critic_target(next_states, next_actions)  # future value
            target_q     = rewards + self.gamma * target_q * (1 - dones)  # Bellman equation

        current_q  = self.critic(states, actions)                  # what critic currently predicts
        critic_loss = nn.MSELoss()(current_q, target_q)            # mean squared error

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()  # negative = maximize

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor,  self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return actor_loss.item(), critic_loss.item()

    def _hard_copy(self, source, target):
        target.load_state_dict(source.state_dict())

    def _soft_update(self, source, target):
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(self.tau * src_param.data + (1 - self.tau) * tgt_param.data)