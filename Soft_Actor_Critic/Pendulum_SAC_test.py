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
sys.path.append("..") # Adds higher directory to python modules path.
from PhyKAN_Util_v2 import PhyKAN_v2
from PhyKAN_Util_v2_SAC import PhyKAN_v2_Actor, PhyKAN_v2_SAC, EfficientReplayBuffer
import gymnasium

# Normalise the angular velocity to range (-1, 1)
def state_normalisation(state):
    state[2] /= 8
    return state

device='cpu'
# Sizes of actions/states, and hidden nodes
Nactions = 1
Nstates = 3

# Instantiate replay buffer
replay_buffer = EfficientReplayBuffer(Nstates, Nactions)

# Load filter values
low_pass_lookup = np.load('../LowFilterVals.npy')
high_pass_lookup = np.load('../HighFilterVals.npy')

# Network parameters
Nfilts = 6
Nhidden = 10

# Learning parameters
gamma = 0.99
critic_lr = 3e-4     
actor_lr = 3e-4      
alpha_lr = 3e-4      
update_tau = 5e-3    
Nepisodes = 500      
batch_size = 64      


# Instantiate SAC networks
agent = PhyKAN_v2_SAC([Nstates, Nhidden, Nhidden, Nactions], Nfilts, low_pass_lookup, high_pass_lookup, gamma=gamma, critic_lr=critic_lr, actor_lr=actor_lr, update_tau=update_tau)

# Instantiate environments
env = gymnasium.make("Pendulum-v1")

for episode in range(Nepisodes):
    state, info = env.reset()
    state = state_normalisation(state)
    episode_reward = 0
    for step in range(200):
        # Retrieve action stochastically from actor
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_t, _ = agent.actor.sample(state_t)
            action = action_t.cpu().squeeze(0).numpy()
        # Step the environment, scaling the squashed action to the range of the task (-2, 2)
        next_state, reward, done, truncated, info = env.step(action*2)
        next_state = state_normalisation(next_state)
        # Store transition in buffer
        replay_buffer.add(state, action, reward, next_state, float(done))
        episode_reward += reward
        state = next_state if not done else env.reset()
        # Perform updates once we have enough samples
        if len(replay_buffer) > batch_size:
            # Sample batch seamlessly as GPU tensors
            b_states, b_actions, b_rewards, b_next_states, b_dones = replay_buffer.sample(batch_size)
            
            # 1. Update Critics
            critic_loss = agent.train_critics(
                states=b_states, 
                actions=b_actions, 
                rewards=b_rewards, 
                next_states=b_next_states, 
                dones=b_dones
            )
            
            # 2. Update Actor (including physics/stability penalties if desired)
            actor_loss, penalty_loss, stability_loss, alpha_loss = agent.train_actor(
                states=b_states, 
                penalty=True, 
                stability_penalty=False
            )
            
            # 3. Soft update target networks toward online critics
            agent.soft_update_targets()
    eval_state, eval_info = env.reset()
    eval_state = state_normalisation(eval_state)
    eval_reward = 0
    for eval_step in range(200):
        with torch.no_grad():
            eval_state_t = torch.FloatTensor(eval_state).unsqueeze(0).to(device)
            # Utilise the deterministic=True flag for stable evaluation
            eval_action_t, _ = agent.actor.sample(eval_state_t, deterministic=True)
            eval_action = eval_action_t.cpu().squeeze(0).numpy()
            
        # Step the environment, scaling the action exactly like the training run
        eval_next_state, eval_reward_step, eval_done, eval_truncated, eval_info = env.step(eval_action * 2)
        eval_next_state = state_normalisation(eval_next_state)
        eval_reward += eval_reward_step
        
        if eval_done or eval_truncated:
            break
        eval_state = eval_next_state
    # ==========================================
    
    print(' ')
    print('Episode: ' + str(episode))
    print('Train Reward: ' + str(episode_reward))
    print('Eval Reward (Deterministic): ' + str(eval_reward))

#%%    
    
# Instantiate a rendering environment
visual_env = gymnasium.make("Pendulum-v1", render_mode="human")

state, info = visual_env.reset()
state = state_normalisation(state)
total_eval_reward = 0

for step in range(200):
    # Render the frame to your display window
    visual_env.render()
    
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_t, _ = agent.actor.sample(state_t, deterministic=True)
        action = action_t.cpu().squeeze(0).numpy()
        
    state, reward, done, truncated, info = visual_env.step(action * 2)
    state = state_normalisation(state)
    total_eval_reward += reward
    
    if done or truncated:
        break

print(f"Visualized Run Completed. Score: {total_eval_reward}")
visual_env.close()















