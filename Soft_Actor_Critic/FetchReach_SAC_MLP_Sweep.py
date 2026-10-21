#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plain MLP-based Soft Actor-Critic test for FetchReachDense-v4.
This mirrors the structure of the existing PhyKAN SAC script but uses
standard fully connected networks for the actor and critics.
"""


import os
import sys
from pathlib import Path

import gymnasium
import gymnasium_robotics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from tqdm import tqdm
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from MLP_SAC_Util import *


device = torch.device("cuda")

from PhyKAN_Util_v2_SAC import EfficientReplayBuffer
if device == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")




def get_goal_conditioned_obs(state_dict):
    return np.concatenate([state_dict["observation"], state_dict["desired_goal"]])

# Network parameters
Nfilts = 6

# Learning parameters
gamma = 0.99
learning_rate = 3e-4         
update_tau = 5e-3    
Nepisodes = 10000    
steps_per_episode = 50
batch_size = 256  

# Network parameters
Nactions = 4
Nstates = 13
replay_buffer = EfficientReplayBuffer(Nstates, Nactions, device=device)
inp = int(sys.argv[1])

Nhiddens = [10, 20, 30, 40, 50, 60, 70, 80, 100]
Nhidden = Nhiddens[inp%9]
for run in range(10):
    agent = MLP_SAC_Agent(
        state_dim=Nstates,
        action_dim=Nactions,
        hidden_dim=Nhidden,
        lr=learning_rate,
        gamma=0.99,
        tau=5e-3,
    ).to(device)
    
    env = gymnasium.make("FetchReachDense-v4")
    
    for episode in tqdm(range(Nepisodes), desc="Episodes"):
        reward_out=[]
        state, _ = env.reset()
        obs = get_goal_conditioned_obs(state)
        episode_reward = 0.0
    
        for step in range(steps_per_episode):
            state_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_t, _ = agent.actor.sample(state_t)
            action = action_t.cpu().squeeze(0).numpy()
    
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_obs = get_goal_conditioned_obs(next_state)
    
            replay_buffer.add(obs, action, reward, next_obs, float(terminated))
            episode_reward += reward
            obs = next_obs
    
            if terminated or truncated:
                break
    
            if len(replay_buffer) > batch_size:
                b_states, b_actions, b_rewards, b_next_states, b_dones = replay_buffer.sample(batch_size)
                agent.train_critics(b_states, b_actions, b_rewards, b_next_states, b_dones)
                agent.train_actor(b_states)
                agent.soft_update_targets()
        reward_out.append(episode_reward)
        if episode % 100 == 0:
            eval_state, _ = env.reset()
            eval_obs = get_goal_conditioned_obs(eval_state)
            eval_reward = 0.0
            for _ in range(steps_per_episode):
                state_t = torch.as_tensor(eval_obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    eval_action_t, _ = agent.actor.sample(state_t, deterministic=True)
                eval_action = eval_action_t.cpu().squeeze(0).numpy()
    
                eval_next_state, eval_reward_step, terminated, truncated, _ = env.step(eval_action)
                eval_reward += eval_reward_step
                eval_obs = get_goal_conditioned_obs(eval_next_state)
                if terminated or truncated:
                    break
    
            print(f"Episode {episode}: train_reward={episode_reward:.2f}, eval_reward={eval_reward:.2f}")
    
            os.makedirs("FetchReach_N_Sweep_Data", exist_ok=True)
            save_path = Path("FetchReach_N_Sweep_Data") / f"FetchReach_SAC_MLP_Nhidden{Nhidden}_run{run}"
            torch.save(agent, save_path)
            print(f"Saved agent to {save_path}")
            np.save(os.getcwd()+f'/FetchReach_N_Sweep_Data/FetchReach_SAC_Rewards_MLP_Nhidden{Nhidden}Run{run}', reward_out)
    
    
