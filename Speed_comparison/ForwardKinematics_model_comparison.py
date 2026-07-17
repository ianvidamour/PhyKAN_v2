#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 12:17:04 2024

@author: ian
"""
import numpy as np
import torch
import torch.nn as nn
import time
device='cuda'
torch.set_default_device(device)
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from PhyKAN_Util import PhyKAN
from PhyKAN_Util_v2 import PhyKAN_v2

def gen_samples(Xin, Yin, batch_size):
    indices = torch.randint(len(Xin), (batch_size,), device=device)
    Xsample = Xin[indices]
    Ysample = Yin[indices]
    return Xsample, Ysample

low_vals = np.load('LowFilterVals.npy')
high_vals = np.load('HighFilterVals.npy')

N = 50
nfilt = 6

Ntr = 100
accuracies = []
Nprune = 2000

Xdata = np.load('6a_joint_angles.npy')
Ydata = np.load('6a_effector_locations.npy')
Xin, x_test  = np.split(Xdata, [15000])
Yin, y_test  = np.split(Ydata, [15000])
idxs = np.arange(0, len(Xin), 1, dtype='int')

np.random.shuffle(idxs)
shuffled_X = Xin[idxs]
shuffled_Y = Yin[idxs]

x_train, x_val = np.split(shuffled_X, [12000])
x_train = torch.from_numpy(x_train).float().to(device)
x_val = torch.from_numpy(x_val).float().to(device)

y_train, y_val = np.split(shuffled_Y, [12000])
y_train = torch.from_numpy(y_train).float().to(device)
y_val = torch.from_numpy(y_val).float().to(device)

shape = [6, N, 3]
model = PhyKAN(shape, nfilt, low_vals, high_vals, lr=1e-2)
model_v2 = PhyKAN_v2(shape, nfilt, low_vals, high_vals, lr=1e-2)

### Testing with No penalty terms, discrete ###
t1 = time.time()
for i in range(Ntr):
    Xs, Ys = gen_samples(x_train, y_train, 500)
    loss = model.train(Xs, Ys, penalty=False, stability_penalty=False, discrete=True)
t2 = time.time()
for i in range(Ntr):
   Xs, Ys = gen_samples(x_train, y_train, 500)
   loss = model_v2.train(Xs, Ys, penalty=False, stability_penalty=False, discrete=True)
t3 = time.time()
v1_time = t2 - t1
v2_time = t3 - t2
print('  ')
print('Discrete, no penalties')
print('V1 time: '+str(v1_time))
print('V2 time: '+str(v2_time))
print('Improvement factor: '+str(v1_time/v2_time))
print('  ')
### Testing with regularisation penalty terms, discrete ###
t1 = time.time()
for i in range(Ntr):
   Xs, Ys = gen_samples(x_train, y_train, 500)
   loss = model.train(Xs, Ys, penalty=True, stability_penalty=False, discrete=True)
t2 = time.time()
for i in range(Ntr):
   Xs, Ys = gen_samples(x_train, y_train, 500)
   loss = model_v2.train(Xs, Ys, penalty=True, stability_penalty=False, discrete=True)
t3 = time.time()
v1_time = t2 - t1
v2_time = t3 - t2
print('  ')
print('Discrete, Regularisation penalties only')
print('V1 time: '+str(v1_time))
print('V2 time: '+str(v2_time))
print('Improvement factor: '+str(v1_time/v2_time))
print('  ')

### Testing with all penalty terms, discrete ###
t1 = time.time()
for i in range(Ntr):
   Xs, Ys = gen_samples(x_train, y_train, 500)
   loss = model.train(Xs, Ys, penalty=True, stability_penalty=True, discrete=True)
t2 = time.time()
for i in range(Ntr):
   Xs, Ys = gen_samples(x_train, y_train, 500)
   loss = model_v2.train(Xs, Ys, penalty=True, stability_penalty=True, discrete=True)
t3 = time.time()
v1_time = t2 - t1
v2_time = t3 - t2
print('  ')
print('Discrete, All penalties')
print('V1 time: '+str(v1_time))
print('V2 time: '+str(v2_time))
print('Improvement factor: '+str(v1_time/v2_time))
print('  ')

### Testing with No penalty terms, continuous ###
t1 = time.time()
for i in range(Ntr):
    Xs, Ys = gen_samples(x_train, y_train, 500)
    loss = model.train(Xs, Ys, penalty=False, stability_penalty=False, discrete=False)
t2 = time.time()
for i in range(Ntr):
   Xs, Ys = gen_samples(x_train, y_train, 500)
   loss = model_v2.train(Xs, Ys, penalty=False, stability_penalty=False, discrete=False)
t3 = time.time()
v1_time = t2 - t1
v2_time = t3 - t2
print('  ')
print('Continuous, no penalties')
print('V1 time: '+str(v1_time))
print('V2 time: '+str(v2_time))
print('Improvement factor: '+str(v1_time/v2_time))
print('  ')

### Testing with regularisation penalty terms, continuous ###
t1 = time.time()
for i in range(Ntr):
   Xs, Ys = gen_samples(x_train, y_train, 500)
   loss = model.train(Xs, Ys, penalty=True, stability_penalty=False, discrete=False)
t2 = time.time()
for i in range(Ntr):
   Xs, Ys = gen_samples(x_train, y_train, 500)
   loss = model_v2.train(Xs, Ys, penalty=True, stability_penalty=False, discrete=False)
t3 = time.time()
v1_time = t2 - t1
v2_time = t3 - t2
print('  ')
print('Continuous, Regularisation penalties only')
print('V1 time: '+str(v1_time))
print('V2 time: '+str(v2_time))
print('Improvement factor: '+str(v1_time/v2_time))
print('  ')

### Testing with all penalty terms, continuous ###
t1 = time.time()
for i in range(Ntr):
   Xs, Ys = gen_samples(x_train, y_train, 500)
   loss = model.train(Xs, Ys, penalty=True, stability_penalty=True, discrete=False)
t2 = time.time()
for i in range(Ntr):
   Xs, Ys = gen_samples(x_train, y_train, 500)
   loss = model_v2.train(Xs, Ys, penalty=True, stability_penalty=True, discrete=False)
t3 = time.time()
v1_time = t2 - t1
v2_time = t3 - t2
print('  ')
print('Continuous, All penalties')
print('V1 time: '+str(v1_time))
print('V2 time: '+str(v2_time))
print('Improvement factor: '+str(v1_time/v2_time))
print('  ')
