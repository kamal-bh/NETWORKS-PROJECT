import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class Actor(nn.Module):
    def __init__(self, state_dim=8, action_dim=8, hidden=128):
        super(Actor, self).__init__()

        # Base QoS priority logits corresponding to DiffServ hierarchy
        self.register_buffer("base_logits", torch.tensor([2.2, 1.8, 1.5, 1.2, 0.9, 0.6, 0.3, 0.0]))
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        # Dynamically modulates base QoS weights by up to ±1.2 logits based on live queue backlogs
        delta = self.network(state) * 1.2
        return torch.softmax(self.base_logits + delta, dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim=8, action_dim=8, hidden=128):
        super(Critic, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class ReplayBuffer:
    def __init__(self, capacity=20_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class OUNoise:
    def __init__(self, action_dim=8, mu=0.0, theta=0.15, sigma=0.15):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state = self.state + dx
        return self.state


class DDPGAgent:
    def __init__(
        self,
        state_dim=8,
        action_dim=8,
        hidden=128,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.005,
        buffer_capacity=20_000,
        batch_size=64,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, hidden).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden).to(self.device)

        self._hard_copy(self.actor, self.actor_target)
        self._hard_copy(self.critic, self.critic_target)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = ReplayBuffer(buffer_capacity)
        self.noise = OUNoise(action_dim)

    def select_action(self, state, add_noise=True, noise_scale=1.0):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_t).squeeze(0).cpu().numpy()
        self.actor.train()

        if add_noise:
            noise = self.noise.sample() * noise_scale
            action = action + noise

        # Ensure minimal floor bandwidth (1.5%) so low-priority queues never starve on bursts
        action = np.clip(action, 0.015, None)
        return action / action.sum()

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None, None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * target_q * (1 - dones)

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5.0)
        self.critic_optimizer.step()

        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return actor_loss.item(), critic_loss.item()

    def _hard_copy(self, source, target):
        target.load_state_dict(source.state_dict())

    def _soft_update(self, source, target):
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(self.tau * src_param.data + (1 - self.tau) * tgt_param.data)

    def save(self, filepath="ddpg_checkpoint.pth"):
        """Save model weights to disk."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }, filepath)

    def load(self, filepath="ddpg_checkpoint.pth"):
        """Load model weights from disk if file exists."""
        import os
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.actor.load_state_dict(checkpoint["actor"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.actor_target.load_state_dict(checkpoint["actor_target"])
            self.critic_target.load_state_dict(checkpoint["critic_target"])
            return True
        return False