import numpy as np
import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F
import sys

sys.path.append("..") # Adds higher directory to python modules path.
from PhyKAN_Util_v2 import PhyKAN_v2


class PhyKAN_v2_Actor(PhyKAN_v2):
    """
    Subclass of PhyKAN_v2 that implements a dual-headed policy network
    producing independent Mean and Log-Std outputs.
    """
    def __init__(self, KANshape, filters_per_unit, low_pass_lookup, high_pass_lookup, **kwargs):
        # Find the shape up until the two different output heads.
        shared_shape = KANshape[:-1]
        # Build the KAN up to the layer preceeding output heads
        super().__init__(shared_shape, filters_per_unit, low_pass_lookup, high_pass_lookup, **kwargs)
        # Find shape of the output heads
        in_dim = KANshape[-2]
        out_dim = KANshape[-1]
        # Build the mean head parameters as in inherited class.
        low_params = torch.linspace(-1, 1, filters_per_unit).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        layer_params = torch.tile(low_params, (in_dim, out_dim, 1, 2))
        noise = (torch.rand(layer_params.shape) - 0.5) * 0.01
        factor = np.sqrt(6 / (filters_per_unit * (in_dim + out_dim)))
        gains_mean = (2 * torch.rand((in_dim, out_dim, filters_per_unit, 1)) - 1) * factor
        # Add the mean head
        self.mean_head_params = nn.Parameter(torch.cat((gains_mean, layer_params + noise), dim=3))
        self.register_buffer('edge_mask_mean', torch.ones((in_dim, out_dim)))
        # Add the standard deviation head
        noise = (torch.rand(layer_params.shape) - 0.5) * 0.01
        gains_std = (2 * torch.rand((in_dim, out_dim, filters_per_unit, 1)) - 1) * factor
        self.logstd_head_params = nn.Parameter(torch.cat((gains_std, layer_params + noise), dim=3))
        self.register_buffer('edge_mask_std', torch.ones((in_dim, out_dim)))

    # Forward pass through the actor network
    def forward(self, Xin, discrete=False, return_edges=False):
        edge_activations = []
        node_activations = []
        # Bound inputs to [0,1]
        current_input = self.sig(Xin)
        for layer in range(self.Nlayers):
            # Get pruning mask from the registry buffer
            current_mask = getattr(self, f'edge_mask_{layer}')
            # Hardware discretisation
            if discrete:
                edges, nodes = self.band_pass_discrete(current_input, self.filter_params[layer], current_mask)
            else:
                edges, nodes = self.band_pass_continuous(current_input, self.filter_params[layer], current_mask)
            # Add activatoins on each edge and each node to output lists
            edge_activations.append(edges)
            node_activations.append(nodes)
            # For all layers except outputs apply sigmoid conditioning step to maintain physically meaningful freq encoding
            if layer < self.Nlayers - 1:
                current_input = self.sig(nodes)
        # Take the activations at the end of the shared layers
        shared_output = self.sig(node_activations[-1])
        # Get pruning masks for the output heads
        mean_mask = getattr(self, 'edge_mask_mean')
        std_mask = getattr(self, 'edge_mask_std')
        # Hardware aware discretisation
        if discrete:
            mean_edges, mean_nodes = self.band_pass_discrete(shared_output, self.mean_head_params, mean_mask)
            std_edges, std_nodes = self.band_pass_discrete(shared_output, self.logstd_head_params, std_mask)
        else:
            mean_edges, mean_nodes = self.band_pass_continuous(shared_output, self.mean_head_params, mean_mask)
            std_edges, std_nodes = self.band_pass_continuous(shared_output, self.logstd_head_params, std_mask)
        # Add the outputs of each head to the activation lists
        edge_activations.extend([mean_edges, std_edges])
        node_activations.extend([mean_nodes, std_nodes])
        # Return outputs according to flags
        if not return_edges:
            return node_activations
        else:
            return node_activations, edge_activations
    
    # Gaussian sampling according to outputs of mean and standard deviation heads
    def sample(self, state, discrete=False, deterministic=False, return_edges=False):
        # Optional return of just node activations or activations + edges for penalty terms
        if return_edges:
            node_activations, edge_activations = self.forward(state, discrete=discrete, return_edges=True)
        else:
            node_activations = self.forward(state, discrete=discrete, return_edges=False)
            edge_activations = None
        # Take means and standard deviations from the different heads
        means = node_activations[-2]
        logstds = node_activations[-1]
        
        # When running deterministically, take only the mean values and convert to actions via tanh squashing
        if deterministic:
            actions = torch.tanh(means)
            if return_edges:
                return actions, None, edge_activations
            return actions, None
        # When training, use gaussian sampling
        
        # Clamp the standard deviations to avoid NaN errors and excessive exploration
        stds = torch.exp(torch.clamp(logstds, -10, 2))
        # Sample using reparameterisation trick
        dist = distributions.Normal(means, stds)
        samples = dist.rsample()
        # Squash samples using tanh function
        actions = torch.tanh(samples)
        # Get log probabilities for entropy term
        log_prob = dist.log_prob(samples)
        log_prob -= torch.log(1.0 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        # Return outputs
        if return_edges:
            return actions, log_prob, edge_activations
        return actions, log_prob

# Whole class that calls the actor internally and builds the four critic networks
class PhyKAN_v2_SAC(nn.Module):
    def __init__(self, KANshape, filters_per_unit, low_pass_lookup, high_pass_lookup, 
                 lr_decay=0.99, sigmoid_s=0.5, actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, 
                 gamma=0.99, update_tau=0.005, filter_alpha=3.65, filter_beta=1.9, 
                 encode_alpha=3.7, encode_beta=1.5):
        super().__init__()
        # Store parameters for discount factor and slow averaging of target critics.
        self.gamma = gamma
        self.update_tau = update_tau
        # Build the actor
        self.actor = PhyKAN_v2_Actor(
            KANshape=KANshape,
            filters_per_unit=filters_per_unit,
            low_pass_lookup=low_pass_lookup,
            high_pass_lookup=high_pass_lookup,
            lr_decay=lr_decay,
            sigmoid_s=sigmoid_s,
            filter_alpha=filter_alpha,
            filter_beta=filter_beta,
            encode_alpha=encode_alpha,
            encode_beta=encode_beta
        )
        # Find dimensions of observation space and action space
        state_dim = KANshape[0]
        action_dim = KANshape[-1]
        critic_input_dim = state_dim + action_dim
        # Define the critic shape to give a single Q value for state-action pair
        critic_shape = [critic_input_dim] + list(KANshape[1:-1]) + [1]
        # Build double networks for fast-adapting critics (gradient based learning)
        self.critic_1 = PhyKAN_v2(critic_shape, filters_per_unit, low_pass_lookup, high_pass_lookup, lr=critic_lr, lr_decay=lr_decay, sigmoid_s=sigmoid_s, filter_alpha=filter_alpha, filter_beta=filter_beta, encode_alpha=encode_alpha, encode_beta=encode_beta)
        self.critic_2 = PhyKAN_v2(critic_shape, filters_per_unit, low_pass_lookup, high_pass_lookup, lr=critic_lr, lr_decay=lr_decay, sigmoid_s=sigmoid_s, filter_alpha=filter_alpha, filter_beta=filter_beta, encode_alpha=encode_alpha, encode_beta=encode_beta)
        # Build double networks for slow-adapting critics (target prediction)
        self.critic_target_1 = PhyKAN_v2(critic_shape, filters_per_unit, low_pass_lookup, high_pass_lookup, lr=critic_lr, lr_decay=lr_decay, sigmoid_s=sigmoid_s, filter_alpha=filter_alpha, filter_beta=filter_beta, encode_alpha=encode_alpha, encode_beta=encode_beta)
        self.critic_target_2 = PhyKAN_v2(critic_shape, filters_per_unit, low_pass_lookup, high_pass_lookup, lr=critic_lr, lr_decay=lr_decay, sigmoid_s=sigmoid_s, filter_alpha=filter_alpha, filter_beta=filter_beta, encode_alpha=encode_alpha, encode_beta=encode_beta)
        # Copy the initialisation parameters for the two fast moving networks to the two slow moving networks to start at the same point
        self.hard_update_targets()
        # Heuristic definition of target entropy as the negative of the action dimension.
        self.target_entropy = -action_dim
        # Parameter controlling scaling of the entropy term, starting high initially
        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True))
        # Build optimiser for the actor network
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # Combine the parameter lists of both gradient updated critic networks to single optimiser for joint optimisation
        critic_params = list(self.critic_1.parameters()) + list(self.critic_2.parameters())
        # Build optimiser for gradient based critic networks
        self.critic_optim = torch.optim.Adam(critic_params, lr=critic_lr)
        # Build optimiser for scaling the entropy term
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    # Get q values from the gradient-based critic networks
    def get_q_values(self, states, actions, discrete=False):
        inputs = torch.cat([states, actions], dim=-1)
        q1 = self.critic_1(inputs, discrete=discrete)[-1]
        q2 = self.critic_2(inputs, discrete=discrete)[-1]
        return q1, q2
    
    # Get q values for the slow-moving critic networks
    def get_target_q_values(self, states, actions, discrete=False):
        inputs = torch.cat([states, actions], dim=-1)
        q1 = self.critic_target_1(inputs, discrete=discrete)[-1]
        q2 = self.critic_target_2(inputs, discrete=discrete)[-1]
        return q1, q2

    # Training loop for updating the gradient-based critics
    def train_critics(self, states, actions, rewards, next_states, dones, discrete=False):
        # Clear gradient information
        self.critic_optim.zero_grad()
        # Forward pass through the actor networks, skipping the gradient accumulation for speed as not used for updating the critic networks
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states, discrete=discrete)
            target_q1, target_q2 = self.get_target_q_values(next_states, next_actions, discrete=discrete)
            min_target_q = torch.minimum(target_q1, target_q2)
            alpha = self.log_alpha.exp()
            target_q = rewards + self.gamma * (1.0 - dones) * (min_target_q - alpha * next_log_probs)
        # Get q values from state-action pairs
        current_q1, current_q2 = self.get_q_values(states, actions, discrete=discrete)
        # Find loss for each of the networks
        critic_loss_1 = F.mse_loss(current_q1, target_q)
        critic_loss_2 = F.mse_loss(current_q2, target_q)
        # Combine losses for joint optimisation
        joint_loss = critic_loss_1 + critic_loss_2
        # Backward pass
        joint_loss.backward()
        # Step critic networks
        self.critic_optim.step()
        # Return joint loss for tracking learning
        return joint_loss.item()

    # Training loop for the actor network
    def train_actor(self, states, penalty=True, stability_penalty=True, lamda=1e-4, stablam=1e-5, discrete=False):
        # Clear gradients
        self.actor_optim.zero_grad()
        # Flag for adding finding edge activations if needed for penalty calculation
        if penalty:
            actions, log_probs, edge_activations = self.actor.sample(states, discrete=discrete, return_edges=True)
        else:
            actions, log_probs = self.actor.sample(states, discrete=discrete, return_edges=False)
        # Get q values from fast critics
        q1, q2 = self.get_q_values(states, actions, discrete=discrete)
        # Take the lowest predicted q values to avoid over optimistic estimation
        min_q = torch.minimum(q1, q2)
        # Exponential scaling of stored log value
        alpha = self.log_alpha.exp()
        # Scale entropy term, ignoring gradients with respect to the alpha parameter.
        actor_loss = (alpha.detach() * log_probs - min_q).mean()
        # Calculate contribution from regularisation penalty terms.
        if penalty:
            penalty_loss = lamda * self.actor.penalty(edge_activations)
            total_loss = actor_loss + penalty_loss
        else:
            total_loss = actor_loss
            penalty_loss = torch.tensor(0.0, device=states.device)
        # Placeholder for stability loss
        stability_loss = torch.tensor(0.0, device=states.device)
        # If using stability penalty, calculate contribution
        if stability_penalty:
            actor_params = list(self.actor.parameters())
            grads = torch.autograd.grad(total_loss, actor_params, create_graph=True)
            for g in grads:
                if g is not None:
                    stability_loss = stability_loss + g.abs().sum() * stablam
            # Add all losses before backward pass
            final_loss = total_loss + stability_loss
            final_loss.backward()
            # Take the loss without the stability penalty if flag is false
        else:
            total_loss.backward()
        # Step actor parameters
        self.actor_optim.step()
        # Clear gradients for the alpha parameter
        self.alpha_optim.zero_grad()
        # Calculate loss between current alpha parameter and target entropy 
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        # Loss for the entropy parameter
        alpha_loss.backward()
        # Update entropy parameter
        self.alpha_optim.step()
        # Return losses for tracking
        return actor_loss.item(), penalty_loss.item(), stability_loss.item(), alpha_loss.item()
    
    # Function to copy parameters upon initialisation
    def hard_update_targets(self):
        with torch.no_grad():
            for target_param, online_param in zip(self.critic_target_1.parameters(), self.critic_1.parameters()):
                target_param.data.copy_(online_param.data)
            for target_param, online_param in zip(self.critic_target_2.parameters(), self.critic_2.parameters()):
                target_param.data.copy_(online_param.data)
                
    # Function for soft averaging of slow moving prediction networks to converge towards gradient-based critics
    def soft_update_targets(self):
        with torch.no_grad():
            for target_param, online_param in zip(self.critic_target_1.parameters(), self.critic_1.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.update_tau) + online_param.data * self.update_tau)
            for target_param, online_param in zip(self.critic_target_2.parameters(), self.critic_2.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.update_tau) + online_param.data * self.update_tau)

# Class for managing experience replay buffer
class EfficientReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=1_000_000, device="cpu"):
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)

        states = torch.from_numpy(self.states[indices]).to(self.device)
        actions = torch.from_numpy(self.actions[indices]).to(self.device)
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device)
        next_states = torch.from_numpy(self.next_states[indices]).to(self.device)
        dones = torch.from_numpy(self.dones[indices]).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size