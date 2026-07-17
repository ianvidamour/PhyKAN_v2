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
        # We initialise the base PhyKAN using ONLY the shared representation layers.
        # This keeps self.filter_params dedicated strictly to the shared trunk.
        shared_shape = KANshape[:-1]
        super().__init__(shared_shape, filters_per_unit, low_pass_lookup, high_pass_lookup, **kwargs)
        
        # Dimensions of the shared head output and target action space
        in_dim = KANshape[-2]
        out_dim = KANshape[-1]
        
        # Initialise Mean Head Parameters
        low_params = torch.linspace(-1, 1, filters_per_unit).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        layer_params = torch.tile(low_params, (in_dim, out_dim, 1, 2))
        noise = (torch.rand(layer_params.shape) - 0.5) * 0.01
        factor = np.sqrt(6 / (filters_per_unit * (in_dim + out_dim)))
        gains_mean = (2 * torch.rand((in_dim, out_dim, filters_per_unit, 1)) - 1) * factor
        
        self.mean_head_params = nn.Parameter(torch.cat((gains_mean, layer_params + noise), dim=3))
        self.register_buffer('edge_mask_mean', torch.ones((in_dim, out_dim)))
        
        # Initialise Log-Std Head Parameters (independent of mean parameters)
        noise = (torch.rand(layer_params.shape) - 0.5) * 0.01
        gains_std = (2 * torch.rand((in_dim, out_dim, filters_per_unit, 1)) - 1) * factor
        self.logstd_head_params = nn.Parameter(torch.cat((gains_std, layer_params + noise), dim=3))
        self.register_buffer('edge_mask_std', torch.ones((in_dim, out_dim)))

    def forward(self, Xin, discrete=False, return_edges=False):
        # Initialise output lists
        edge_activations = []
        node_activations = []
        
        # Forward through the shared layers
        current_input = self.sig(Xin)
        for layer in range(self.Nlayers):
            current_mask = getattr(self, f'edge_mask_{layer}')
            if discrete:
                edges, nodes = self.band_pass_discrete(current_input, self.filter_params[layer], current_mask)
            else:
                edges, nodes = self.band_pass_continuous(current_input, self.filter_params[layer], current_mask)
            
            edge_activations.append(edges)
            node_activations.append(nodes)
            if layer < self.Nlayers - 1:
                current_input = self.sig(nodes)
        
        # Output of the shared trunk
        shared_output = self.sig(node_activations[-1])
        
        # Forward through both head parameters
        mean_mask = getattr(self, 'edge_mask_mean')
        std_mask = getattr(self, 'edge_mask_std')
        
        if discrete:
            mean_edges, mean_nodes = self.band_pass_discrete(shared_output, self.mean_head_params, mean_mask)
            std_edges, std_nodes = self.band_pass_discrete(shared_output, self.logstd_head_params, std_mask)
        else:
            mean_edges, mean_nodes = self.band_pass_continuous(shared_output, self.mean_head_params, mean_mask)
            std_edges, std_nodes = self.band_pass_continuous(shared_output, self.logstd_head_params, std_mask)
            
        # Here, the outputs of the mean head will be the second to last in the list, with stds last
        edge_activations.extend([mean_edges, std_edges])
        node_activations.extend([mean_nodes, std_nodes])
        
        if not return_edges:
            return node_activations
        else:
            return node_activations, edge_activations

    def sample(self, state, discrete=False, deterministic=False):
        # 1. Pass through the model to get means and logstds
        node_activations = self.forward(state, discrete=discrete, return_edges=False)
        means = node_activations[-2]
        logstds = node_activations[-1]
        
        if deterministic:
            # For evaluation/deployment:
            # Bypass sampling completely and directly squash the mean.
            # No log_probs are needed for evaluation.
            actions = torch.tanh(means)
            return actions, None
        
        # For training:
        # Calculate standard deviation and sample stochastically
        stds = torch.exp(torch.clamp(logstds, -10, 2))
        dist = distributions.Normal(means, stds)
        samples = dist.rsample()  # Reparameterization trick
        actions = torch.tanh(samples)
        
        # Calculate log probabilities and apply the stable tanh correction
        log_prob = dist.log_prob(samples)
        tanh_correction = 2.0 * (0.69314718056 - samples - F.softplus(-2.0 * samples))
        log_prob -= tanh_correction
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return actions, log_prob


class PhyKAN_v2_SAC(nn.Module):
    """
    Unified Soft Actor-Critic (SAC) framework that manages both Actor 
    and Twin Critic PhyKANs, their respective target networks, and optimisation.
    """
    def __init__(self, KANshape, filters_per_unit, low_pass_lookup, high_pass_lookup, 
                 lr_decay=0.99, sigmoid_s=0.5, actor_lr=1e-3, critic_lr=1e-3, alpha_lr=1e-4, 
                 gamma=0.99, update_tau=0.001, filter_alpha=3.65, filter_beta=1.9, 
                 encode_alpha=3.7, encode_beta=1.5):
        super().__init__()
        
        self.gamma = gamma
        self.update_tau = update_tau
        
        # Instantiate the single Actor instance
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
        
        # Determine and instantiate Twin Critics & Targets
        state_dim = KANshape[0]
        action_dim = KANshape[-1]
        critic_input_dim = state_dim + action_dim
        
        # Auto-compute critic shape: [state + action, hidden_layers..., 1]
        critic_shape = [critic_input_dim] + list(KANshape[1:-1]) + [1]
        
        # Online Critics
        self.critic_1 = PhyKAN_v2(critic_shape, filters_per_unit, low_pass_lookup, high_pass_lookup, lr=critic_lr, lr_decay=lr_decay, sigmoid_s=sigmoid_s, filter_alpha=filter_alpha, filter_beta=filter_beta, encode_alpha=encode_alpha, encode_beta=encode_beta)
        self.critic_2 = PhyKAN_v2(critic_shape, filters_per_unit, low_pass_lookup, high_pass_lookup, lr=critic_lr, lr_decay=lr_decay, sigmoid_s=sigmoid_s, filter_alpha=filter_alpha, filter_beta=filter_beta, encode_alpha=encode_alpha, encode_beta=encode_beta)
        
        # Target Critics (slow update networks)
        self.critic_target_1 = PhyKAN_v2(critic_shape, filters_per_unit, low_pass_lookup, high_pass_lookup, lr=critic_lr, lr_decay=lr_decay, sigmoid_s=sigmoid_s, filter_alpha=filter_alpha, filter_beta=filter_beta, encode_alpha=encode_alpha, encode_beta=encode_beta)
        self.critic_target_2 = PhyKAN_v2(critic_shape, filters_per_unit, low_pass_lookup, high_pass_lookup, lr=critic_lr, lr_decay=lr_decay, sigmoid_s=sigmoid_s, filter_alpha=filter_alpha, filter_beta=filter_beta, encode_alpha=encode_alpha, encode_beta=encode_beta)
        
        # Hard copy the initialised online parameters directly to target parameters
        self.hard_update_targets()
        
        # Dynamic Alpha setup
        self.target_entropy = -action_dim
        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True))
        
        # Policy optimiser (automatically optimises shared + head parameters)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # Combined critics optimiser
        critic_params = list(self.critic_1.parameters()) + list(self.critic_2.parameters())
        self.critic_optim = torch.optim.Adam(critic_params, lr=critic_lr)
        # Alpha optimiser
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    def get_q_values(self, states, actions, discrete=False):
        # Function to get q values from both online critic networks
        inputs = torch.cat([states, actions], dim=-1)
        q1 = self.critic_1(inputs, discrete=discrete)[-1]
        q2 = self.critic_2(inputs, discrete=discrete)[-1]
        return q1, q2

    def get_target_q_values(self, states, actions, discrete=False):
        # Function to get target q values from both soft updated networks
        inputs = torch.cat([states, actions], dim=-1)
        q1 = self.critic_target_1(inputs, discrete=discrete)[-1]
        q2 = self.critic_target_2(inputs, discrete=discrete)[-1]
        return q1, q2

    def train_critics(self, states, actions, rewards, next_states, dones, discrete=False):
        # Training loop for the online critic networks
        self.critic_optim.zero_grad()
        # Pass through policy networks for actions, and target q networks with no gradient tracking
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states, discrete=discrete)
            target_q1, target_q2 = self.get_target_q_values(next_states, next_actions, discrete=discrete)
            min_target_q = torch.minimum(target_q1, target_q2)
            alpha = self.log_alpha.exp()
            # Bellman equation with entropy term added
            target_q = rewards + self.gamma * (1.0 - dones) * (min_target_q - alpha * next_log_probs)
            
        current_q1, current_q2 = self.get_q_values(states, actions, discrete=discrete)
        
        critic_loss_1 = F.mse_loss(current_q1, target_q)
        critic_loss_2 = F.mse_loss(current_q2, target_q)
        joint_loss = critic_loss_1 + critic_loss_2
        
        joint_loss.backward()
        self.critic_optim.step()
        
        return joint_loss.item()

    def train_actor(self, states, penalty=True, stability_penalty=True, lamda=1e-4, stablam=1e-5, discrete=False):
        # Training loop for the actor networks
        self.actor_optim.zero_grad()
        
        # Retrieve edge activations for the actor's physics penalty if required
        if penalty:
            node_activations, edge_activations = self.actor.forward(states, discrete=discrete, return_edges=True)
            actions, log_probs = self.actor.sample(states, discrete=discrete)
        else:
            actions, log_probs = self.actor.sample(states, discrete=discrete)
            
        q1, q2 = self.get_q_values(states, actions, discrete=discrete)
        min_q = torch.minimum(q1, q2)
        
        alpha = self.log_alpha.exp()
        # Actor loss (minimisation of: alpha * log_prob - min_q)
        actor_loss = (alpha.detach() * log_probs - min_q).mean()
        
        # Add KAN penalisation
        if penalty:
            penalty_loss = lamda * self.actor.penalty(edge_activations)
            total_loss = actor_loss + penalty_loss
        else:
            total_loss = actor_loss
            penalty_loss = torch.tensor(0.0, device=states.device)
            
        stability_loss = torch.tensor(0.0, device=states.device)
        
        # Add stability penalty for transferred actor networks
        if stability_penalty:
            # Dual-backprop through autograd to compute the gradient norm penalty
            actor_params = list(self.actor.parameters())
            grads = torch.autograd.grad(total_loss, actor_params, create_graph=True)
            for g in grads:
                if g is not None:
                    stability_loss = stability_loss + g.abs().sum() * stablam
            
            final_loss = total_loss + stability_loss
            final_loss.backward()
        else:
            total_loss.backward()
            
        self.actor_optim.step()
        
        # Update Temperature Parameter (Alpha)
        self.alpha_optim.zero_grad()
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.alpha_optim.step()
        
        return actor_loss.item(), penalty_loss.item(), stability_loss.item(), alpha_loss.item()

    # Hard deep copy of parameters used during initialisation
    def hard_update_targets(self):
        with torch.no_grad():
            for target_param, online_param in zip(self.critic_target_1.parameters(), self.critic_1.parameters()):
                target_param.data.copy_(online_param.data)
            for target_param, online_param in zip(self.critic_target_2.parameters(), self.critic_2.parameters()):
                target_param.data.copy_(online_param.data)

    def soft_update_targets(self):
        # Soft update for target networks
        with torch.no_grad():
            for target_param, online_param in zip(self.critic_target_1.parameters(), self.critic_1.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.update_tau) + online_param.data * self.update_tau)
            for target_param, online_param in zip(self.critic_target_2.parameters(), self.critic_2.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.update_tau) + online_param.data * self.update_tau)
    
class EfficientReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=1_000_000, device="cpu"):
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0

        # Pre-allocate flat memory buffers (using NumPy for fast CPU storage)
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """
        Inserts a single step transition into the buffer in O(1) time.
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        # Advance circular pointer, wrap around if buffer is full
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
        Samples a batch of transitions and directly outputs PyTorch tensors 
        on the target device.
        """
        # Vectorized random choice of indices
        indices = np.random.randint(0, self.size, size=batch_size)

        # Vectorized array slicing + instant PyTorch tensor conversion
        # torch.from_numpy shares memory with numpy, meaning zero-copy tensor creation!
        states = torch.from_numpy(self.states[indices]).to(self.device)
        actions = torch.from_numpy(self.actions[indices]).to(self.device)
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device)
        next_states = torch.from_numpy(self.next_states[indices]).to(self.device)
        dones = torch.from_numpy(self.dones[indices]).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size