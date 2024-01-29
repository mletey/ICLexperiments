"""
Can these models learn regression in-context? We're about to find out!

original author: William Tong (wtong@g.harvard.edu)
contributor: Mary Letey for 2024 In Context paper
"""

# <codecell>
import jax
from flax.serialization import from_state_dict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import sys
sys.path.append('../../')
sys.path.append('../../../')
from common import *
from train import train, create_train_state
from model.knn import KnnConfig
from model.mlp import MlpConfig
from model.poly import PolyConfig
from model.transformer import TransformerConfig
from task.regression import LinearRegression

d=50;
tvals = np.array([i/10 for i in range(1,20,2)]); Pvals = d**2.*tvals;
alpha = 5; N = int(alpha*d);

sigma = 0.1;
psi = 1;

i = int(sys.argv[1]); # grab value of $SLURM_ARRAY_TASK_ID to index over taus
P = int(Pvals[i]);

linobject = LinearRegression(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, batch_size = P, seed=None);

config = TransformerConfig(pos_emb=False, n_hidden=256, max_len=1000) 
state, hist = train(config, data_iter=iter(linobject), loss='mse', test_every=1000, train_iters=200_000, lr=1e-4)

xs, ys = next(linobject); # generates data
y_pred = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
err = np.mean((ys - y_pred) ** 2);

file_path = f"result{i}.txt"
with open(file_path, 'w') as file:
    file.write(f"tau is {tvals[i]}\n")
    file.write(f"error is {err}\n")

