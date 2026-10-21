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
from tqdm import tqdm
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from MLP_SAC_Util import MLP_SAC_Agent
from PhyKAN_Util_v2_SAC import EfficientReplayBuffer

device = torch.device("cuda")


if device == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


# Learning parameters
gamma = 0.99
learning_rate = 3e-4         
update_tau = 5e-3    
Nepisodes = 1000    
steps_per_episode = 200
batch_size = 256  

# Network parameters
Nactions = 1
Nstates = 3

for inp in range(9):
    #inp = int(sys.argv[1])
    
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
            hidden_layers=1
        ).to(device)
        replay_buffer = EfficientReplayBuffer(Nstates, Nactions, device=device)
        env = gymnasium.make("Pendulum-v1")
        reward_out=[]
        for episode in tqdm(range(Nepisodes), desc="Episodes"):
            state, _ = env.reset()
            episode_reward = 0.0
        
            for step in range(steps_per_episode):
                state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action_t, _ = agent.actor.sample(state_t)
                action = action_t.cpu().squeeze(0).numpy()
        
                next_state, reward, terminated, truncated, _ = env.step(2.0 * action)
                reward /= 50.0 # Similar scaling as for the KAN to keep things comparable
                replay_buffer.add(state, action, reward, next_state, float(terminated))
                episode_reward += reward
                state = next_state
        
                if terminated or truncated:
                    break
        
                if len(replay_buffer) > batch_size:
                    b_states, b_actions, b_rewards, b_next_states, b_dones = replay_buffer.sample(batch_size)
                    agent.train_critics(b_states, b_actions, b_rewards, b_next_states, b_dones)
                    agent.train_actor(b_states)
                    agent.soft_update_targets()
            eval_state, eval_info = env.reset()
            eval_reward = 0
            if episode%10==0:
                for eval_step in range(steps_per_episode):
                    with torch.no_grad():
                        eval_state_t = torch.FloatTensor(eval_state).unsqueeze(0).to(device)
                        # Utilise the deterministic=True flag for stable evaluation
                        eval_action_t, _ = agent.actor.sample(eval_state_t, deterministic=True)
                        eval_action = eval_action_t.cpu().squeeze(0).numpy()
                        
                    # Step the environment, scaling the action exactly like the training run
                    eval_next_state, eval_reward_step, eval_done, eval_truncated, eval_info = env.step(2.0 * eval_action)
                    eval_reward += eval_reward_step
                    
                    if eval_done or eval_truncated:
                        break
                    eval_state = eval_next_state
                reward_out.append(eval_reward)
        
                print(f"Episode {episode}: train_reward={episode_reward:.2f}, eval_reward={eval_reward:.2f}")
        
                save_path = Path("Pendulum_N_Sweep_Data") / f"Pendulum_SAC_MLP_Nhidden{Nhidden}_run{run}"
                torch.save(agent, save_path)
                print(f"Saved agent to {save_path}")
                np.save(os.getcwd()+f'/Pendulum_N_Sweep_Data/Pendulum_SAC_Rewards_MLP_Nhidden{Nhidden}Run{run}', reward_out)
        
        
