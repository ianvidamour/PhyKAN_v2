#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 16:30:56 2026

@author: ian
"""
# Imports of necessary modules
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F
import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from PhyKAN_Util_v2 import PhyKAN_v2
from PhyKAN_Util_v2_SAC import PhyKAN_v2_Actor, PhyKAN_v2_SAC, EfficientReplayBuffer
import gymnasium
import gymnasium_robotics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Goal-Aware Sizes: 63 (kinematics) + 15 ( fingertip goals) = 78 total features
Nactions = 20
Nstates = 78  

# Instantiate replay buffer
replay_buffer = EfficientReplayBuffer(Nstates, Nactions, device=device)

# Load filter values
low_pass_lookup = np.load('../LowFilterVals.npy')
high_pass_lookup = np.load('../HighFilterVals.npy')

# Network parameters (Here we will sweep Nhidden to find the best architecture for this task)
Nfilts = 6
inp = int(sys.argv[1])
Nhiddens = [16, 32, 64, 128, 256]
Nhidden = Nhiddens[inp]

# Learning parameters
gamma = 0.99
critic_lr = 3e-4     
actor_lr = 3e-4      
alpha_lr = 3e-4      
update_tau = 5e-3    
Nepisodes = 30000    
batch_size = 256  

# Instantiate SAC networks
agent = PhyKAN_v2_SAC([Nstates, Nhidden, Nhidden, Nactions], Nfilts, low_pass_lookup, high_pass_lookup, gamma=gamma, critic_lr=critic_lr, actor_lr=actor_lr, update_tau=update_tau)
agent.to(device)

# Instantiate environments
env = gymnasium.make("HandReachDense-v3")

# Helper function to extract and concatenate goal-aware states
def get_goal_conditioned_obs(state_dict):
    return np.concatenate([state_dict["observation"], state_dict["desired_goal"]])


 
for episode in range(Nepisodes):
    state, info = env.reset()
    # Concatenate observation and target goal 
    obs = get_goal_conditioned_obs(state)
    episode_reward = 0
    
    for step in range(50):
        # Retrieve action stochastically from actor
        with torch.no_grad():
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action_t, _ = agent.actor.sample(state_t)
            action = action_t.cpu().squeeze(0).numpy()
            
        # Step the environment using the squashed action directly (maps perfectly to [-1, 1])
        next_state, reward, terminated, truncated, info = env.step(action)
        next_obs = get_goal_conditioned_obs(next_state)
        
        # Store goal-conditioned transition in buffer
        replay_buffer.add(obs, action, reward, next_obs, float(terminated))
        episode_reward += reward
        obs = next_obs
        
        if terminated or truncated:
            break
            
        # Perform updates once we have enough samples
        if len(replay_buffer) > batch_size:
            b_states, b_actions, b_rewards, b_next_states, b_dones = replay_buffer.sample(batch_size)
            
            # 1. Update Critics
            critic_loss = agent.train_critics(
                states=b_states, 
                actions=b_actions, 
                rewards=b_rewards, 
                next_states=b_next_states, 
                dones=b_dones
            )
            
            # 2. Update Actor
            actor_loss, penalty_loss, stability_loss, alpha_loss = agent.train_actor(
                states=b_states, 
                penalty=True, 
                stability_penalty=False
            )
            
            # 3. Soft update target networks
            agent.soft_update_targets()
    if episode % 100 == 0:
        # ==========================================
        # Evaluation Run (Deterministic & Goal-Conditioned)
        # ==========================================
        eval_state, eval_info = env.reset()
        eval_obs = get_goal_conditioned_obs(eval_state)
        eval_reward = 0
        
        for eval_step in range(50):
            with torch.no_grad():
                eval_state_t = torch.FloatTensor(eval_obs).unsqueeze(0).to(device)
                eval_action_t, _ = agent.actor.sample(eval_state_t, deterministic=True)
                eval_action = eval_action_t.cpu().squeeze(0).numpy()
                
            eval_next_state, eval_reward_step, eval_terminated, eval_truncated, eval_info = env.step(eval_action)
            eval_reward += eval_reward_step
            
            if eval_terminated or eval_truncated:
                break
            eval_obs = get_goal_conditioned_obs(eval_next_state)
        # ==========================================
        
        print(' ')
        print('Episode: ' + str(episode))
        print('Train Reward: ' + str(episode_reward))
        print('Eval Reward (Deterministic): ' + str(eval_reward))

# Save the models after training
torch.save(agent, f'../Reach_N_Sweep_Data/Reach_SAC_Actor_Nhidden{Nhidden}')
