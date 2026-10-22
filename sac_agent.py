import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import random
from collections import deque


class SACActorNetwork(nn.Module):
    def __init__(self, state_dim=8, action_dim=8, hidden=128, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_dim = action_dim

        # Base QoS priority logits corresponding to DiffServ hierarchy
        self.register_buffer("base_logits", torch.tensor([2.2, 1.8, 1.5, 1.2, 0.9, 0.6, 0.3, 0.0]))
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

        with torch.no_grad():
            self.mean_head.weight.data.mul_(0.01)
            self.mean_head.bias.data.zero_()
            self.log_std_head.weight.data.mul_(0.01)
            self.log_std_head.bias.data.fill_(-2.0)

    def forward(self, state):
        x = self.trunk(state)
        mean = self.mean_head(x)
        log_std = torch.clamp(self.log_std_head(x), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()

        # Dynamic residual modulation around base QoS logits
        delta = torch.tanh(z) * 1.2
        det_delta = torch.tanh(mean) * 1.2

        action = F.softmax(self.base_logits + delta, dim=-1)
        det_action = F.softmax(self.base_logits + det_delta, dim=-1)

        # Log prob of Gaussian sample for entropy regularization
        log_prob = dist.log_prob(z).sum(dim=-1, keepdim=True)

        return action, log_prob, det_action


class SACCriticNetwork(nn.Module):
    def __init__(self, state_dim=8, action_dim=8, hidden=128):
        super().__init__()
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


class SACReplayBuffer:
    def __init__(self, capacity=20_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)),
            torch.FloatTensor(np.array(a)),
            torch.FloatTensor(np.array(r)).unsqueeze(1),
            torch.FloatTensor(np.array(ns)),
            torch.FloatTensor(np.array(d)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class SACAgent:
    def __init__(
        self,
        state_dim=8,
        action_dim=8,
        hidden=128,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=64,
        buffer_capacity=20_000,
        target_entropy=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = SACActorNetwork(state_dim, action_dim, hidden).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic1 = SACCriticNetwork(state_dim, action_dim, hidden).to(self.device)
        self.critic2 = SACCriticNetwork(state_dim, action_dim, hidden).to(self.device)
        self.critic1_target = SACCriticNetwork(state_dim, action_dim, hidden).to(self.device)
        self.critic2_target = SACCriticNetwork(state_dim, action_dim, hidden).to(self.device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.c1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.c2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        # Target entropy: calibrated for continuous bounded QoS weights
        self.target_entropy = target_entropy if target_entropy is not None else -float(action_dim)
        # Initialize alpha to ~0.03 to balance with the scaled environment step rewards (~ -0.35)
        self.log_alpha = torch.tensor(-3.5, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr * 0.5)

        self.buffer = SACReplayBuffer(buffer_capacity)

    def select_action(self, state, deterministic=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _, det_action = self.actor.sample(state_t)
            if deterministic:
                act = det_action.squeeze(0).cpu().numpy()
            else:
                act = action.squeeze(0).cpu().numpy()

        # Ensure minimal floor bandwidth (1.5%) so low-priority queues never starve on bursts
        act = np.clip(act, 0.015, None)
        return act / act.sum()

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None, None, None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)
            target_q = rewards + self.gamma * (1 - dones) * (q_next - self.alpha * next_log_probs)

        # Critic 1
        q1_pred = self.critic1(states, actions)
        c1_loss = F.mse_loss(q1_pred, target_q)
        self.c1_optimizer.zero_grad()
        c1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=5.0)
        self.c1_optimizer.step()

        # Critic 2
        q2_pred = self.critic2(states, actions)
        c2_loss = F.mse_loss(q2_pred, target_q)
        self.c2_optimizer.zero_grad()
        c2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=5.0)
        self.c2_optimizer.step()

        # Actor
        new_actions, log_probs, _ = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
        self.actor_optimizer.step()

        # Alpha temperature
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=2.0)
        self.alpha_optimizer.step()

        # Clamp log_alpha so temperature stays well-balanced with reward scale
        with torch.no_grad():
            self.log_alpha.clamp_(-6.0, -2.0)
            self.alpha = self.log_alpha.exp().item()

        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        return actor_loss.item(), (c1_loss.item() + c2_loss.item()) / 2, self.alpha

    def _soft_update(self, source, target):
        for s_p, t_p in zip(source.parameters(), target.parameters()):
            t_p.data.copy_(self.tau * s_p.data + (1 - self.tau) * t_p.data)

    def save(self, filepath="sac_checkpoint.pth"):
        """Save model weights to disk."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "log_alpha": self.log_alpha.data,
        }, filepath)

    def load(self, filepath="sac_checkpoint.pth"):
        """Load model weights from disk if file exists."""
        import os
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.actor.load_state_dict(checkpoint["actor"])
            self.critic1.load_state_dict(checkpoint["critic1"])
            self.critic2.load_state_dict(checkpoint["critic2"])
            self.critic1_target.load_state_dict(checkpoint["critic1_target"])
            self.critic2_target.load_state_dict(checkpoint["critic2_target"])
            if "log_alpha" in checkpoint:
                self.log_alpha.data.copy_(checkpoint["log_alpha"])
                self.alpha = self.log_alpha.exp().item()
            return True
        return False