#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 15:02:08 2025

@author: ian
"""

import numpy as np
import torch
import torch.nn as nn

class PhyKAN(nn.Module):
    def __init__(self, KANshape, filters_per_unit, low_pass_lookup, high_pass_lookup, thresholding=False, lr_decay=0.99, sigmoid_s=0.5, lr=1e-3):
        super().__init__()
        # Find number of layers
        self.Nlayers = len(KANshape)-1
        self.KANshape = KANshape
        # Create parameters for each layer
        self.filter_params = []
        self.thresholds = []
        self.edge_mask = []
        for i in range(self.Nlayers):
            low_params = torch.linspace(-1, 1, filters_per_unit).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            layer_params = torch.tile(low_params, (KANshape[i], KANshape[i+1], 1, 2))
            noise = (torch.rand(layer_params.shape)-0.5)*0.01
            layer_params = layer_params + noise
            gains = torch.rand((KANshape[i], KANshape[i+1], filters_per_unit, 1))/filters_per_unit
            params = torch.cat((gains, layer_params), dim=3)
            params.requires_grad=True
            self.filter_params.append(params)
            layer_mask = torch.ones((KANshape[i], KANshape[i+1]))
            self.edge_mask.append(layer_mask)
            if thresholding==True:
                thresholds = 0.25*torch.randn(KANshape[i])
                thresholds.requires_grad=True
            else:
                thresholds = torch.zeros(KANshape[i], requires_grad=True)
            self.thresholds.append(thresholds)
        # Initialise optimisers
        self.filter_optim = torch.optim.Adam(params=self.filter_params, lr=lr)
        self.threshold_optim = torch.optim.Adam(params=self.thresholds, lr=1e-5)
        self.sched1 = torch.optim.lr_scheduler.ExponentialLR(self.filter_optim, 0.99)
        self.lossfn = nn.MSELoss()
        self.thresholding=thresholding
        self.sigmoid_s = sigmoid_s
        self.low_pass_lookup = torch.tensor(low_pass_lookup)
        self.high_pass_lookup = torch.tensor(high_pass_lookup)
    
    def sig(self, x):
        return 1/(1+torch.exp(-x/self.sigmoid_s))
        
    # Redefined for torch & assuming trainable parameters
    def band_pass_discrete(self, Xin, params, mask):
        # Bound parameters
        params= self.sig(params)
        # Encode inputs to frequency space
        prefreq = 10**(3.65+1.5*Xin).unsqueeze(-1).unsqueeze(-1)
        freq = (4e6/np.pi)*torch.arctan((np.pi*prefreq)/4e6)
        # Allow bounded gains to be both positive and negative
        gain = 3*(params[:,:,:, 0].unsqueeze(0)-0.5)
        gain[gain==0]=0.01
        # Pass through discretisation to maintain backprop (straight-through estimator)
        discretised_gain = gain + gain.round(decimals=2).detach() - gain.detach()
        gain[gain==0]=0.01
        # Set cutoff frequencies from parameters
        fc_low = 10**(3.65+1.9*params[:,:,:, 1])
        fc_high = 10**(3.65+1.9*params[:,:,:, 2])
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
        for i in range(fc_low.shape[0]):
            for j in range(fc_low.shape[1]):
                for k in range(fc_low.shape[2]):
                    discretised_fc_low[i,j,k] = self.low_pass_lookup[mindiff_low[i,j,k]]
                    discretised_fc_high[i,j,k] = self.high_pass_lookup[mindiff_high[i,j,k]]
        # Passthrough discretisation to enable backprop (straight-through estimator)
        passed_fc_low = fc_low + discretised_fc_low.detach() - fc_low.detach()
        passed_fc_high = fc_high + discretised_fc_high.detach() - fc_high.detach()
        # Convert cutoff frequency to product of resistance and capacitance
        RC_low = (2*torch.pi*passed_fc_low.unsqueeze(0))**-1
        RC_high = (2*torch.pi*passed_fc_high.unsqueeze(0))**-1
        # Calculate steady state response with respect to drive frequency
        Hout = discretised_gain*torch.abs((2j*torch.pi*freq*RC_high/(1+2j*torch.pi*freq*RC_high))*(1/(1+2j*torch.pi*freq*RC_low)))
        # Apply mask
        Hout = Hout * mask.unsqueeze(0).unsqueeze(-1)
        # Sum across units within each edge, and across multiple inputs to each node
        return Hout.sum(dim=1).sum(dim=-1)
    
    def band_pass_continuous(self, Xin, params, mask):
        # Bound parameters
        params= self.sig(params)
        # Encode inputs to frequency space
        prefreq = 10**(3.65+1.5*Xin).unsqueeze(-1).unsqueeze(-1)
        freq = (4e6/np.pi)*torch.arctan((np.pi*prefreq)/4e6)
        # Allow bounded gains to be both positive and negative
        gain = 3*(params[:,:,:, 0].unsqueeze(0)-0.5)
        gain[gain==0]=0.01
        # Set cutoff frequencies from parameters
        fc_low = 10**(3.65+1.9*params[:,:,:, 1])
        fc_high = 10**(3.65+1.9*params[:,:,:, 2])
        # Convert cutoff frequency to product of resistance and capacitance
        RC_low = (2*torch.pi*fc_low.unsqueeze(0))**-1
        RC_high = (2*torch.pi*fc_high.unsqueeze(0))**-1
        # Calculate steady state response with respect to drive frequency
        Hout = gain*torch.abs((2j*torch.pi*freq*RC_high/(1+2j*torch.pi*freq*RC_high))*(1/(1+2j*torch.pi*freq*RC_low)))
        # Apply mask
        Hout = Hout * mask.unsqueeze(0).unsqueeze(-1)
        # Sum across units within each edge, and across multiple inputs to each node
        return Hout.sum(dim=1).sum(dim=-1)

    def band_pass_edges(self, Xin, params):
        # Bound parameters
        params= self.sig(params)
        # Encode inputs to frequency space
        prefreq = 10**(3.65+1.5*Xin).unsqueeze(-1).unsqueeze(-1)
        freq = (4e6/np.pi)*torch.arctan((np.pi*prefreq)/4e6)
        # Allow bounded gains to be both positive and negative
        gain = 3*(params[:, 0].unsqueeze(0)-0.5)
        # Set cutoff frequencies from parameters
        fc_low = 10**(3.65+1.9*params[:, 1].unsqueeze(0))
        fc_high = 10**(3.65+1.9*params[:, 2].unsqueeze(0))
        # Convert cutoff frequency to product of resistance and capacitance
        RC_low = (2*torch.pi*fc_low)**-1
        RC_high = (2*torch.pi*fc_high)**-1
        # Calculate steady state response with respect to drive frequency
        Hout = gain*torch.abs((2j*torch.pi*freq*RC_high/(1+2j*torch.pi*freq*RC_high))*(1/(1+2j*torch.pi*freq*RC_low)))
        # Sum across units within each edge
        return Hout.sum(dim=-1)
    
    # Redefined for torch & assuming trainable parameters
    def band_pass_single(self, Xin, params, threshold):
        # Threshold input
        Xin = (Xin-threshold).relu()
        # Bound parameters
        params=self.sig(params)
        # Normalise input of 0-1 to frequencies of 100 Hz-100 kHz
        prefreq = 10**(3.65+1.5*Xin).unsqueeze(-1).unsqueeze(-1)
        freq = (4e6/np.pi)*torch.arctan((np.pi*prefreq)/4e6)
        # For 'sensible' initialisation, we'd like parameters that make sense for gain, f_low, and f_high:
        # Set gain between +/- 5X
        gain = 3*(params[:, 0].unsqueeze(0)-0.5)
        # Set cutoff frequencies between 10 Hz and 1 MHz
        fc_low = 10**(3.65+1.9*params[:, 1].unsqueeze(0))
        fc_high = 10**(3.65+1.9*params[:, 2].unsqueeze(0))
        RC_low = (2*torch.pi*fc_low)**-1
        RC_high = (2*torch.pi*fc_high)**-1
        Hout = gain*torch.abs((2j*torch.pi*freq*RC_high/(1+2j*torch.pi*freq*RC_high))*(1/(1+2j*torch.pi*freq*RC_low)))
        return Hout.sum(dim=1).sum(dim=-1)
    
    def forward(self, Xin, discrete=False):
        batch_size = len(Xin)
        # Initialise inputs to and outputs from each layer
        inputs = []
        outputs = []
        for i in range(self.Nlayers):
            inputs.append(torch.zeros((batch_size, self.KANshape[i])))
            outputs.append(torch.zeros((batch_size, self.KANshape[i+1])))
        if self.thresholding==True:
            inputs[0][:, :] = self.sig((Xin-self.thresholds[0]))
        else:
            inputs[0][:,:] = self.sig((Xin))
        if discrete==True:
            # Pass through model
            for layer in range(self.Nlayers):
                outputs[layer][:, :] = self.band_pass_discrete(inputs[layer], self.filter_params[layer], self.edge_mask[layer])
                if layer < self.Nlayers-1:
                    if self.thresholding==True:
                        inputs[layer+1][:, :] = self.sig(outputs[layer][:, :]+self.thresholds[layer+1])
                    else:
                        inputs[layer+1][:, :] = self.sig(outputs[layer][:, :])
        else:
            # Pass through model
            for layer in range(self.Nlayers):
                outputs[layer][:, :] = self.band_pass_continuous(inputs[layer], self.filter_params[layer], self.edge_mask[layer])
                if layer < self.Nlayers-1:
                    if self.thresholding==True:
                        inputs[layer+1][:, :] = self.sig(outputs[layer][:, :]+self.thresholds[layer+1])
                    else:
                        inputs[layer+1][:, :] = self.sig(outputs[layer][:, :])
        return outputs
    
    def train(self, Xin, Yin, penalty=True, stability_penalty=True, lamda=1e-4, stablam=1e-5, discrete=False):
        # Reset optimisers
        self.filter_optim.zero_grad()
        self.threshold_optim.zero_grad()
        # Pass through Model
        activations = self.forward(Xin, discrete=discrete)
        pred = activations[-1]
        # Calculate loss
        loss_np = self.lossfn(pred, Yin)
        # Add penalty terms
        if penalty==True:
            penalty_loss = lamda * self.penalty(activations)
            loss = loss_np + penalty_loss
        else:
            loss = loss_np
        # Backward pass
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
        self.threshold_optim.step()
        
        return loss.item(), loss_np.item(), stability_loss.item()

    def penalty(self, activations):
        n = len(activations[0])
        # Set up variable
        penalty = 0
        # Loop over layers
        for layer in range(self.Nlayers-1):
            # Calculate l1 penalties 
            l1_in = (1/n) * activations[layer].abs().sum(dim=0)/self.KANshape[layer]
            l1_out = (1/n) * activations[layer+1].abs().sum(dim=0)/self.KANshape[layer+1]
            l1_layer = l1_in.sum() + l1_out.sum()
            # Calculate entropy penalties (ReLU/ small addition is to prevent NaNs pruned edges)
            entropy_in = -1 * (l1_in/l1_layer) * torch.log(1e-9+ (l1_in/l1_layer).relu())
            entropy_out = -1 * (l1_out/l1_layer) * torch.log(1e-9 + (l1_out/l1_layer).relu())
            entropy_layer = (entropy_in.sum() + entropy_out.sum())
            # Combine terms
            penalty = penalty + l1_layer + entropy_layer
        return penalty
        
    def prune(self, threshold = 0.01):
        xs = torch.rand(10000, self.KANshape[0])
        outputs = self.forward_return_acts(xs)
        for layer in range(self.Nlayers-1):
            layerouts = torch.abs(outputs[layer]).mean(axis=0)
            for node, activation in enumerate(layerouts):
                if activation < threshold:
                    self.reshape(layer, node)

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
            
    def reshape(self, layer, node):
        newshape = []
        for l in range(self.Nlayers+1):
            if l != layer+1:
                newshape.append(self.KANshape[l])
            else:
                newshape.append(self.KANshape[l]-1)
        new_params = []
        flag = 0
        for l in range(self.Nlayers):
            if flag == 0:
                if l != layer:
                    new_params.append(self.filter_params[l])
                else:
                    layer_params=[]
                    for i in range(self.KANshape[l+1]):
                        if i != node:
                            layer_params.append(self.filter_params[l][:, i, :, :])
                    layer_params = torch.stack(layer_params, dim=1)
                    new_params.append(layer_params)
                    flag = 1
            else:
                layer_params=[]
                for i in range(self.KANshape[l]):
                    if i != node:
                        layer_params.append(self.filter_params[l][i, :, :, :])
                layer_params = torch.stack(layer_params, dim=0)
                new_params.append(layer_params)
                flag = 0
        self.KANshape = newshape
        self.filter_params = new_params
        print('Pruned: Layer '+str(layer)+', Node '+str(node))
        print('New Shape: '+str(self.KANshape))
                
