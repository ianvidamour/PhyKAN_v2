#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 15:02:08 2026

@author: ian
"""

import numpy as np
import torch
import torch.nn as nn

class PhyKAN_v2(nn.Module):
    def __init__(self, KANshape, filters_per_unit, low_pass_lookup, high_pass_lookup, lr_decay=0.99, sigmoid_s=0.5, lr=1e-3, filter_alpha=3.65, filter_beta=1.9, encode_alpha=3.7, encode_beta=1.5):
        super().__init__()
        # Find number of layers
        self.Nlayers = len(KANshape)-1
        self.KANshape = KANshape
        # Set encoding parameters
        self.filter_alpha = filter_alpha
        self.filter_beta = filter_beta
        self.encode_alpha = encode_alpha
        self.encode_beta = encode_beta
        
        # Properly register parameters using nn.ParameterList
        self.filter_params = nn.ParameterList()
        for i in range(self.Nlayers):
            low_params = torch.linspace(-1, 1, filters_per_unit).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            layer_params = torch.tile(low_params, (KANshape[i], KANshape[i+1], 1, 2))
            # Noise filter parameters so not identical on all edges
            noise = (torch.rand(layer_params.shape)-0.5)*0.01
            layer_params = layer_params + noise
            # Xavier-esque initialisation factor for gains
            factor = np.sqrt(6 / (filters_per_unit * (KANshape[i] + KANshape[i+1])))
            gains = (2 * torch.rand((KANshape[i], KANshape[i+1], filters_per_unit, 1)) - 1) * factor
            params = torch.cat((gains, layer_params), dim=3)
            
            # Wrap in nn.Parameter so PyTorch tracks gradients
            self.filter_params.append(nn.Parameter(params))
            layer_mask = torch.ones((KANshape[i], KANshape[i+1]))
            self.register_buffer(f'edge_mask_{i}', layer_mask)
            
        # Initialise optimisers
        self.filter_optim = torch.optim.Adam(params=self.filter_params, lr=lr)
        self.sched1 = torch.optim.lr_scheduler.ExponentialLR(self.filter_optim, lr_decay)
        self.lossfn = nn.MSELoss()
        self.sigmoid_s = sigmoid_s
        
        # Register lookups as buffers to ensure auto-GPU migration
        self.register_buffer('low_pass_lookup', torch.tensor(low_pass_lookup, dtype=torch.float32))
        self.register_buffer('high_pass_lookup', torch.tensor(high_pass_lookup, dtype=torch.float32))
    
    def sig(self, x):
        # Fix: Changed to lowercase 'sigmoid' to avoid class instantiation error
        return torch.sigmoid(x/self.sigmoid_s)
        
    def band_pass_discrete(self, Xin, params, mask):
        # Bound parameters
        bounded_params = self.sig(params)
        # Encode inputs to frequency space
        prefreq = torch.pow(10.0, self.encode_alpha + self.encode_beta * Xin).unsqueeze(-1).unsqueeze(-1)
        freq = (4e6/np.pi)*torch.arctan((np.pi*prefreq)/4e6)
        # Allow bounded gains to be both positive and negative
        gain = 3*(bounded_params[:,:,:, 0].unsqueeze(0)-0.5)
        gain[gain==0]=0.01
        # Pass through discretisation to maintain backprop (straight-through estimator)
        discretised_gain = gain + gain.round(decimals=2).detach() - gain.detach()
        # Set cutoff frequencies from parameters
        fc_low = 10**(self.filter_alpha + self.filter_beta * bounded_params[:, :, :, 1])
        fc_high = 10**(self.filter_alpha + self.filter_beta * bounded_params[:, :, :, 2])
        
        # Find difference between values via broadcasting
        difference_low = (self.low_pass_lookup - fc_low.unsqueeze(-1)).abs()
        difference_high = (self.high_pass_lookup - fc_high.unsqueeze(-1)).abs()
        
        # Find minima along varied axis
        mindiff_low = torch.argmin(difference_low, dim=-1)
        mindiff_high = torch.argmin(difference_high, dim=-1)
        
        # Select discretised samples
        discretised_fc_low = self.low_pass_lookup[mindiff_low]
        discretised_fc_high = self.high_pass_lookup[mindiff_high]
        
        # Passthrough discretisation to enable backprop (straight-through estimator)
        passed_fc_low = fc_low + discretised_fc_low.detach() - fc_low.detach()
        passed_fc_high = fc_high + discretised_fc_high.detach() - fc_high.detach()
        
        # Fix: Use passed_fc_* variables so that backward gradients can bypass detach()
        x = freq / passed_fc_low
        y = freq / passed_fc_high
        
        # Use torch.hypot to safely compute sqrt(1 + x^2) without precision collapse
        ones = torch.ones_like(x)
        mag_high = x / torch.hypot(ones, x)
        mag_low = 1.0 / torch.hypot(ones, y)
        
        # Combine responses
        Hout = discretised_gain * (mag_high * mag_low) * mask.unsqueeze(0).unsqueeze(-1)
        # Return sum over sub-filters for edge activations
        activations = Hout.sum(dim=-1)
        # Return sum over incoming connections for node inputs
        node_inputs = activations.sum(dim=1)
        return activations, node_inputs

    def band_pass_continuous(self, Xin, params, mask):
        # Bound parameters
        bounded_params= self.sig(params)
        # Encode inputs to frequency space
        prefreq = torch.pow(10.0, self.encode_alpha + self.encode_beta * Xin).unsqueeze(-1).unsqueeze(-1)
        freq = (4e6/np.pi)*torch.arctan((np.pi*prefreq)/4e6)
        # Map to filter parameters
        gain = 3*(bounded_params[:,:,:, 0].unsqueeze(0)-0.5)
        fc_low = 10**(self.filter_alpha + self.filter_beta * bounded_params[:, :, :, 1])
        fc_high = 10**(self.filter_alpha + self.filter_beta * bounded_params[:, :, :, 2])
        # Refactored forward pass to avoid complex numbers
        x = freq / fc_low
        y = freq / fc_high
        # Use torch.hypot to safely compute sqrt(1 + x^2) without precision collapse
        ones = torch.ones_like(x)
        mag_high = x / torch.hypot(ones, x)
        mag_low = 1.0 / torch.hypot(ones, y)
        # Combine responses
        Hout = gain * (mag_high * mag_low) * mask.unsqueeze(0).unsqueeze(-1)
        # Return sum over sub-filters for edge activations
        activations = Hout.sum(dim=-1)
        # Return sum over incoming connections for node inputs
        node_inputs = activations.sum(dim=1) 
        return activations, node_inputs

    def forward(self, Xin, discrete=False, return_edges=False):
        # Initialise empty lists to store layer outputs dynamically
        edge_activations = []
        node_activations = []
        # The input to the first layer is the sigmoid of Xin
        current_input = self.sig(Xin)
        # Pass through the model layer by layer
        for layer in range(self.Nlayers):
            # Load the current edge mask for the layer
            current_mask = getattr(self, f'edge_mask_{layer}')
            # Process the current layer based on discrete or continuous flag
            if discrete:
                edges, nodes = self.band_pass_discrete(
                    current_input, 
                    self.filter_params[layer], 
                    current_mask
                )
            else:
                edges, nodes = self.band_pass_continuous(
                    current_input, 
                    self.filter_params[layer], 
                    current_mask
                )
            # Store the resulting tensors
            edge_activations.append(edges)
            node_activations.append(nodes)
            # Prepare the input for the next layer (if this isn't the final layer)
            if layer < self.Nlayers - 1:
                current_input = self.sig(nodes)
        # Return the requested activations
        if not return_edges:
            return node_activations
        else:
            return node_activations, edge_activations
    
    def train(self, Xin=None, Yin=None, penalty=True, stability_penalty=True, lamda=1e-4, stablam=1e-5, discrete=False):
        # nn.Module.train() uses this method name, so we must preserve PyTorch's
        # training-mode switching semantics when called from .eval() / .train().
        if isinstance(Xin, bool) or isinstance(Yin, bool):
            return super().train(Xin if isinstance(Xin, bool) else Yin)
        if Xin is None and Yin is None:
            return super().train(True)

        # Reset optimisers
        self.filter_optim.zero_grad()
        # Pass through Model
        node_activations, edge_activations = self.forward(Xin, discrete=discrete, return_edges=True)
        pred = node_activations[-1]
        # Calculate loss
        loss_np = self.lossfn(pred, Yin)
        
        # Add penalty terms
        if penalty:
            penalty_loss = lamda * self.penalty(edge_activations, eps=lamda)
            loss = loss_np + penalty_loss
        else:
            loss = loss_np
            penalty_loss = torch.tensor(0.0)
            
        stability_loss = torch.tensor(0.0)
        if stability_penalty:
            # True Gradient Penalty (Double Backprop via autograd.grad)
            grads = torch.autograd.grad(loss, self.filter_params, create_graph=True)
            for g in grads:
                stability_loss = stability_loss + g.abs().sum() * stablam
            
            # Combine loss and perform second backward pass with computed gradient penalties
            loss = loss + stability_loss
            loss.backward()
        else:
            # Single backward pass
            loss.backward()
            
        # Update parameters
        self.filter_optim.step()

        return loss_np.item(), penalty_loss.item(), stability_loss.item()

    # Refactored for improved speed
    def penalty(self, activations, eps=1e-8):
        n = len(activations[0])
        penalty = torch.tensor([0.0], device=activations[0].device)

        for layer in range(self.Nlayers - 1):
            layer_acts = activations[layer]
            abs_acts = layer_acts.abs()

            # L1 sparsity term with a tiny stabilising offset.
            l1_layer = (1.0 / n) * abs_acts.sum() + eps

            # Normalise activations safely across the incoming edges.
            denom = abs_acts.sum(dim=(1, 2), keepdim=True)
            denom = torch.clamp(denom, min=eps)
            probs = abs_acts / denom

            # Compute a positive distribution over incoming edges for entropy regularisation.
            q_ih = (1.0 / n) * probs.sum(dim=0)
            entropy_layer = -(q_ih * torch.log(torch.clamp(q_ih, min=eps))).sum()

            penalty = penalty + l1_layer + entropy_layer

        return penalty

    def prune_edges(self, threshold=0.05):
        xs = torch.linspace(0, 1, 1000)
        for layer in range(self.Nlayers-1):
            # Fetch dynamic edge mask
            current_mask = getattr(self, f'edge_mask_{layer}')
            # Expand xs to (1000, KANshape[layer]) to match expected forward batch shape
            xs_expanded = xs.unsqueeze(-1).expand(-1, self.KANshape[layer])
            # Continuous evaluation of bandpass filters over the layer
            layer_nodes, layer_edges = self.band_pass_continuous(
                xs_expanded, 
                self.filter_params[layer], 
                current_mask
            )
            # Loop over all edges in the layer, pruning if activation is below threshold
            for pre in range(self.KANshape[layer]):
                for post in range(self.KANshape[layer+1]):
                    edges = layer_edges[:, pre, post] 
                    if current_mask[pre, post] == 0:
                        continue
                    if torch.abs(edges).mean() < threshold:
                        current_mask[pre, post] = 0
                        print(f'Pruning edge, layer {layer} {pre} {post}')