#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 17:21:57 2026

@author: ian
"""


import gymnasium
import gymnasium_robotics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions


class MLPActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, hidden_layers=2):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        layers = []
        prev_dim = state_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)

        self.log_std_min = -10.0
        self.log_std_max = 2.0

    def forward(self, x):
        features = self.shared_layers(x)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        if deterministic:
            action = torch.tanh(mean)
            return action, None

        std = torch.exp(log_std)
        dist = distributions.Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class MLPCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, hidden_layers=2):
        super().__init__()
        input_dim = state_dim + action_dim
        layers = []
        prev_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.output_layer(self.net(x))


class MLP_SAC_Agent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, gamma=0.99, tau=5e-3, hidden_layers=2):
        super().__init__()
        self.gamma = gamma
        self.tau = tau

        self.actor = MLPActor(state_dim, action_dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers)
        self.critic_1 = MLPCritic(state_dim, action_dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers)
        self.critic_2 = MLPCritic(state_dim, action_dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers)

        self.target_critic_1 = MLPCritic(state_dim, action_dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers)
        self.target_critic_2 = MLPCritic(state_dim, action_dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers)
        self.hard_update_targets()

        self.target_entropy = -action_dim
        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True))

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=lr,
        )
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)

    def hard_update_targets(self):
        with torch.no_grad():
            for target, source in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                target.data.copy_(source.data)
            for target, source in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                target.data.copy_(source.data)

    def soft_update_targets(self):
        with torch.no_grad():
            for target, source in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                target.data.mul_(1.0 - self.tau).add_(source.data, alpha=self.tau)
            for target, source in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                target.data.mul_(1.0 - self.tau).add_(source.data, alpha=self.tau)

    def get_q_values(self, states, actions):
        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        return q1, q2

    def get_target_q_values(self, states, actions):
        q1 = self.target_critic_1(states, actions)
        q2 = self.target_critic_2(states, actions)
        return q1, q2

    def train_critics(self, states, actions, rewards, next_states, dones):
        self.critic_optim.zero_grad()

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1, target_q2 = self.get_target_q_values(next_states, next_actions)
            min_target_q = torch.minimum(target_q1, target_q2)
            alpha = self.log_alpha.exp()
            target_q = rewards + self.gamma * (1.0 - dones) * (min_target_q - alpha * next_log_probs)

        current_q1, current_q2 = self.get_q_values(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        critic_loss.backward()
        self.critic_optim.step()
        return critic_loss.item()

    def train_actor(self, states):
        self.actor_optim.zero_grad()
        actions, log_probs = self.actor.sample(states)
        q1, q2 = self.get_q_values(states, actions)
        min_q = torch.minimum(q1, q2)

        alpha = self.log_alpha.exp()
        actor_loss = (alpha.detach() * log_probs - min_q).mean()
        actor_loss.backward()
        self.actor_optim.step()

        self.alpha_optim.zero_grad()
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.alpha_optim.step()
        return actor_loss.item(), alpha_loss.item()