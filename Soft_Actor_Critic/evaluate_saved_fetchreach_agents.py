#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate saved FetchReach SAC agents over many initial conditions.
The script loads each saved checkpoint from a directory, runs a deterministic
policy rollout for a configurable number of seeds, and reports the mean and
standard deviation of the total reward attained for each model.
"""

# Imports of necessary modules
import sys
from pathlib import Path
import os
import gymnasium
import gymnasium_robotics
import numpy as np
import torch


parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Ensure the custom KAN and MLP modules are importable when loading checkpoints
try:
    import PhyKAN_Util_v2_SAC  # noqa: F401
    import MLP_SAC_Util  # noqa: F401
except Exception:
    pass

# Function to add observation and goals to inputs to models
def get_goal_conditioned_obs(state_dict):
    return np.concatenate([state_dict["observation"], state_dict["desired_goal"]])


# Load 100 random initial conditions, evaluate reward achieved by models
def evaluate_agent(agent, env, num_seeds=100, max_steps=50, device=None):
    rewards = []

    for seed in range(num_seeds):
        state, info = env.reset(seed=seed)
        obs = get_goal_conditioned_obs(state)
        episode_reward = 0.0

        for step in range(max_steps):
            state_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.inference_mode():
                action_t, _ = agent.actor.sample(state_t, deterministic=True)
            action = action_t.cpu().squeeze(0).numpy()

            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            obs = get_goal_conditioned_obs(next_state)

            if terminated or truncated:
                break

        rewards.append(float(episode_reward))

    return np.array(rewards, dtype=np.float32)

def trainables_KAN(Nhidden):
    # Trainables for KAN: 18 per edge, 13 inputs, 4 outputs, 2 hidden layers of N =
    # 18 * (13 * N + N * N + N * 4)
    return 18 * (17 * Nhidden + Nhidden**2)

def trainables_MLP(Nhidden):
    # Trainables for MLP: 13 inputs, 4 outputs, 2 hidden layers of N =
    # 13 * N + N * N + N * 4) + N * 2 + 4
    return (17 * Nhidden + Nhidden**2) + (2 * Nhidden + 4)



# Experiment parameters
directory = os.getcwd()+"/FetchReach_N_Sweep_Data/"
pattern = "*"
num_seeds = 100
num_runs = 5
max_steps = 50
output = "fetchreach_agent_eval_results.csv"

# Load the environment
env = gymnasium.make("FetchReachDense-v4")

# Swept parameters for MLP and KAN
Nhiddens_KAN = [5, 10, 15, 20, 25]
Nhiddens_MLP = [10, 20, 30, 40, 50, 60, 70, 80, 100]

# Output arrays to take evaluated data
KAN_data = np.zeros((len(Nhiddens_KAN), num_runs, num_seeds))
MLP_data = np.zeros((len(Nhiddens_MLP), num_runs, num_seeds))

# Output arrays for trainable parameters
KAN_trainables = np.zeros((len(Nhiddens_KAN)))
MLP_trainables = np.zeros((len(Nhiddens_MLP)))

# Loop over trained KANs
for i, Nhidden in enumerate(Nhiddens_KAN):
    KAN_trainables[i ]= trainables_KAN(Nhidden)
    for run in range(num_runs):
        model = torch.load(directory+f'FetchReach_SAC_Nhidden{Nhidden}Run{run}', weights_only=False, map_location='cpu')
        KAN_data[i, run] = evaluate_agent(model, env)

# loop over trained MLPs
for i, Nhidden in enumerate(Nhiddens_MLP):
    MLP_trainables[i] = trainables_MLP(Nhidden)
    for run in range(num_runs):
        model = torch.load(directory+f'FetchReach_SAC_MLP_Nhidden{Nhidden}_run{run}', weights_only=False, map_location='cpu')
        MLP_data[i, run] = evaluate_agent(model, env)


import matplotlib.pyplot as plt

mean_KAN = KAN_data.mean(axis=-1)
mean_MLP = MLP_data.mean(axis=-1)
fig, ax = plt.subplots()
ax.set_xscale('log')
ax.errorbar(KAN_trainables, mean_KAN.mean(axis=-1), yerr=mean_KAN.std(axis=-1))
ax.errorbar(MLP_trainables, mean_MLP.mean(axis=-1), yerr=mean_MLP.std(axis=-1))
plt.show()






















