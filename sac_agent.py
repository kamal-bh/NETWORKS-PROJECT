import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import random
from collections import deque


class SACActorNetwork(nn.Module):
    def __init__(self, state_dim=2, hidden=64, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared feature extractor
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.mean_head    = nn.Linear(hidden, 1)  
        self.log_std_head = nn.Linear(hidden, 1)  

    def forward(self, state):
        x       = self.trunk(state)
        mean    = self.mean_head(x)
        log_std = torch.clamp(self.log_std_head(x), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        """
        Sample an action using the reparameterization trick.
        Returns: (action [w1,w2], log_prob, deterministic_action [w1,w2])
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        z    = dist.rsample()                 

        w1 = torch.sigmoid(z)                 
        w2 = 1.0 - w1                         
        action = torch.cat([w1, w2], dim=-1)   

        log_prob = dist.log_prob(z)
        log_prob -= torch.log(w1 * w2 + 1e-6)  
        log_prob  = log_prob.sum(dim=-1, keepdim=True) 
        w1_det = torch.sigmoid(mean)
        det_action = torch.cat([w1_det, 1.0 - w1_det], dim=-1)

        return action, log_prob, det_action


class SACCriticNetwork(nn.Module):
    def __init__(self, state_dim=2, action_dim=2, hidden=64):
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
    def __init__(self, capacity=10_000):
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
        state_dim=2,
        action_dim=2,
        hidden=64,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=64,
        buffer_capacity=10_000,
        target_entropy=-1.0,  
    ):
        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size

        self.actor           = SACActorNetwork(state_dim, hidden)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic1 = SACCriticNetwork(state_dim, action_dim, hidden)
        self.critic2 = SACCriticNetwork(state_dim, action_dim, hidden)
        self.critic1_target = SACCriticNetwork(state_dim, action_dim, hidden)
        self.critic2_target = SACCriticNetwork(state_dim, action_dim, hidden)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.c1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.c2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)


        self.target_entropy  = target_entropy
        self.log_alpha       = torch.tensor(0.0, requires_grad=True)
        self.alpha           = self.log_alpha.exp().item()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)


        self.buffer = SACReplayBuffer(buffer_capacity)

    def select_action(self, state, deterministic=False):
        """
        deterministic=False during training  → sample from distribution (explore)
        deterministic=True  during evaluation → use mean of distribution (exploit)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            _, _, det_action = self.actor.sample(state_t)

            if deterministic:
                action = det_action.squeeze(0).numpy()
            else:
                action, _, _ = self.actor.sample(state_t)
                action = action.squeeze(0).numpy()

        # Safety clip + normalize
        action = np.clip(action, 0.01, None)
        action = action / action.sum()
        return action

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None, None, None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)


        with torch.no_grad():
            # Sample next actions from the stochastic actor
            next_actions, next_log_probs, _ = self.actor.sample(next_states)

            # Twin critics on next state — take the MINIMUM (conservative)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next  = torch.min(q1_next, q2_next)

            # SAC Bellman target includes entropy bonus:
            #   target_Q = r + γ * (min_Q_next  -  α * log_π_next)
            #                                     ↑
            #               This term rewards being uncertain (exploring)
            target_q = rewards + self.gamma * (1 - dones) * (q_next - self.alpha * next_log_probs)

  
        q1_pred     = self.critic1(states, actions)
        c1_loss     = F.mse_loss(q1_pred, target_q)
        self.c1_optimizer.zero_grad()
        c1_loss.backward()
        self.c1_optimizer.step()

        q2_pred     = self.critic2(states, actions)
        c2_loss     = F.mse_loss(q2_pred, target_q)
        self.c2_optimizer.zero_grad()
        c2_loss.backward()
        self.c2_optimizer.step()

        new_actions, log_probs, _ = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new  = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()   # sync alpha scalar

        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        return actor_loss.item(), (c1_loss.item() + c2_loss.item()) / 2, self.alpha

    def _soft_update(self, source, target):
        for s_p, t_p in zip(source.parameters(), target.parameters()):
            t_p.data.copy_(self.tau * s_p.data + (1 - self.tau) * t_p.data)