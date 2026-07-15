#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 15:02:08 2025

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
        self.edge_mask = []
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
            # Wrap in nn.Parameter so PyTorch tracks gradeints
            self.filter_params.append(nn.Parameter(params))
            layer_mask = torch.ones((KANshape[i], KANshape[i+1]))
            self.edge_mask.append(layer_mask)
        # Initialise optimisers
        self.filter_optim = torch.optim.Adam(params=self.filter_params, lr=lr)
        self.sched1 = torch.optim.lr_scheduler.ExponentialLR(self.filter_optim, 0.99)
        self.lossfn = nn.MSELoss()
        self.sigmoid_s = sigmoid_s
        self.low_pass_lookup = torch.tensor(low_pass_lookup)
        self.high_pass_lookup = torch.tensor(high_pass_lookup)
    
    def sig(self, x):
        return torch.Sigmoid(x/self.sigmoid_s)
        
    # Redefined for torch & assuming trainable parameters
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
        # Initialise discretised outputs
        discretised_fc_low = torch.zeros_like(fc_low)
        discretised_fc_high = torch.zeros_like(fc_high)
        # Extend tensors to same dimensions as parameters to ease selection
        expanded_low_lookup = torch.tile(self.low_pass_lookup, (fc_low.shape+(1,)))
        expanded_low_values = torch.tile(fc_low.unsqueeze(-1), (1, 1, 1, self.low_pass_lookup.shape[0]))
        expanded_high_lookup = torch.tile(self.high_pass_lookup, (fc_high.shape+(1,)))
        expanded_high_values = torch.tile(fc_high.unsqueeze(-1), (1, 1, 1, self.high_pass_lookup.shape[0]))
        # Find difference between values
        difference_low = (expanded_low_lookup - expanded_low_values).abs()
        difference_high = (expanded_high_lookup - expanded_high_values).abs()
        # Find minima along varied axis
        mindiff_low = torch.argmin(difference_low, dim=-1)
        mindiff_high = torch.argmin(difference_high, dim=-1)
        # Select discretised samples
        discretised_fc_low[:,:,:] = self.low_pass_lookup[mindiff_low[:,:,:]]
        discretised_fc_high[:,:,:] = self.high_pass_lookup[mindiff_high[:,:,:]]
        # Passthrough discretisation to enable backprop (straight-through estimator)
        passed_fc_low = fc_low + discretised_fc_low.detach() - fc_low.detach()
        passed_fc_high = fc_high + discretised_fc_high.detach() - fc_high.detach()
        # Refactored forward pass to avoid complex numbers
        x = freq / discretised_fc_low
        y = freq / discretised_fc_high
        # Use torch.hypot to safely compute sqrt(1 + x^2) without precision collapse
        ones = torch.ones_like(x)
        mag_high = x / torch.hypot(ones, x)
        mag_low = 1.0 / torch.hypot(ones, y)
        # Combine responses
        Hout = discretised_gain * (mag_high * mag_low)
        # Return sum over sub-filters for edge activations
        activations = Hout.sum(dim=-1)
        # Return sum over incoming connections for node inputs
        node_inputs = activations.sum(dim=1)
        return activations, node_inputs

    
    def band_pass_continuous(self, Xin, params, mask):
        # Bound parameters
        bounded_params= self.sig(params)
        # Encode inputs to frequency space
        prefreq = 10**(3.65+1.5*Xin).unsqueeze(-1).unsqueeze(-1)
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
        Hout = gain * (mag_high * mag_low)
        # Return sum over sub-filters for edge activations
        activations = Hout.sum(dim=-1)
        # Return sum over incoming connections for node inputs
        node_inputs = activations.sum(dim=1)
        return activations, node_inputs

    
    def forward(self, Xin, discrete=False, return_edges=False):
        batch_size = len(Xin)
        # Initialise inputs to and outputs from each layer
        inputs = []
        edge_activations = []
        node_activations = []
        for i in range(self.Nlayers):
            inputs.append(torch.zeros((batch_size, self.KANshape[i])))
            edge_activations.append(torch.zeros((batch_size, self.KANshape[i], self.KANshape[i+1])))
            node_activations.append(torch.zeros((batch_size, self.KANshape[i+1])))
        inputs[0][:,:] = self.sig((Xin))
        if discrete==True:
            # Pass through model
            for layer in range(self.Nlayers):
                edge_activations[layer][:, :, :], node_activations[layer][:, :] = self.band_pass_discrete(inputs[layer], self.filter_params[layer], self.edge_mask[layer])
                if layer < self.Nlayers-1:
                    inputs[layer+1][:, :] = self.sig(node_activations[layer][:, :])
        else:
            # Pass through model
            for layer in range(self.Nlayers):
                edge_activations[layer][:, :, :], node_activations[layer][:, :] = self.band_pass_continuous(inputs[layer], self.filter_params[layer], self.edge_mask[layer])
                if layer < self.Nlayers-1:
                    inputs[layer+1][:, :] = self.sig(node_activations[layer][:, :])
        if return_edges==False:
            return node_activations
        else:
            return node_activations, edge_activations
    
    def train(self, Xin, Yin, penalty=True, stability_penalty=True, lamda=1e-4, stablam=1e-5, discrete=False):
        # Reset optimisers
        self.filter_optim.zero_grad()
        self.threshold_optim.zero_grad()
        # Pass through Model
        node_activations, edge_activations = self.forward(Xin, discrete=discrete, return_edges=True)
        pred = node_activations[-1]
        # Calculate loss
        loss_np = self.lossfn(pred, Yin)
        # Add penalty terms
        if penalty==True:
            penalty_loss = lamda * self.penalty(edge_activations)
            loss = loss_np + penalty_loss
        else:
            loss = loss_np
        stability_loss = 0
        if stability_penalty==True:
            loss.backward(retain_graph=True)
            if stability_penalty == True:
                for layer in self.filter_params:
                    params = torch.zeros_like(layer)
                    params[:,:,:,0] = 3*(layer[:,:,:,0]-0.5)
                    params[:,:,:,1:] = 10**(3.65+1.9*layer[:,:,:,1:])
                    gradients = layer.grad.abs().sum()
                    stability_loss = stability_loss+gradients*stablam
                loss = loss + stability_loss
            # Second backward pass to correctly include stability penalty
            loss.backward()
        else:
            # Backward pass
            loss.backward()
        # Update parameters
        self.filter_optim.step()
        self.threshold_optim.step()
        
        return loss_np.item(), penalty_loss.item(), stability_loss.item()

    # Refactored for improved speed
    def penalty(self, activations):
        n = len(activations[0])
        # Set up variable
        penalty = 0
        # Loop over layers
        for layer in range(self.Nlayers-1):
            # Precompute absolute activations as repeated operation
            abs_acts = layer.abs()
            # Calculate l1 penalties 
            l1_layer = (1/n) * abs_acts.sum()
            # Find q
            q_ih = (1/n) * (abs_acts/abs_acts.sum(dim=(1,2)).unsqueeze(-1).unsqueeze(-1)).sum(dim=0)
            entropy_layer = (q_ih * torch.log(q_ih + 1e-9)).sum()
            # Combine terms
            penalty = penalty + l1_layer + entropy_layer
        return penalty
        

    def prune_edges(self, threshold = 0.05):
        xs = torch.linspace(0, 1, 1000)
        for layer in range(self.Nlayers-1):
            for pre in range(self.KANshape[layer]):
                for post in range(self.KANshape[layer+1]):
                    if self.edge_mask[layer][pre, post] == 0:
                        continue
                    edges = self.band_pass_edges(xs, self.filter_params[layer][pre, post])
                    if torch.abs(edges).mean() < threshold:
                        self.edge_mask[layer][pre, post] = 0
                        print('Pruning edge, layer '+str(layer)+' '+str(pre)+' '+str(post))
            
                
