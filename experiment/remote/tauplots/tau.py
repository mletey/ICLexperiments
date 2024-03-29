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
import optax
import pickle
import sys
print("halfway through intro ugh")
sys.path.append('../../')
sys.path.append('../../../')
from common import *
from train import train, create_train_state
from model.knn import KnnConfig
from model.mlp import MlpConfig
from model.poly import PolyConfig
from model.transformer import TransformerConfig
from task.regression import LinearRegression
from task.regression import LinearRegressionCorrect

d=10;
tvals = np.array([0.1,0.2,0.3,0.4,0.55,0.65,0.75,0.85,0.95,1.05,1.15,1.25,1.35,1.45,1.55,1.7,1.8,1.9,2])
Pvals = d**2.*tvals;
alpha = 1; N = int(alpha*d);

sigma = 0.25;
psi = 1;

i = int(sys.argv[1]) - 1; # grab value of $SLURM_ARRAY_TASK_ID to index over taus
P = int(Pvals[i]);
print("P is",P)
print("N is",N)
linobject = LinearRegressionCorrect(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, batch_size = P, seed=None);
config = TransformerConfig(pos_emb=False, n_hidden=512) 

state, hist = train(config, data_iter=iter(linobject), loss='mse', test_every=1000, train_iters=100000, lr=1e-4)

avgerr = 0;
loss_func = optax.squared_error
for _ in range(10):
  xs, labels = next(linobject); # generates data
  logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
  #print("shape of output is", logits.shape)
#file_path = f'../../../../../../Everyone/mletey_128_results/resultsfiner{i}.txt'
  avgerr = avgerr + loss_func(logits, labels).mean()

avgerr = avgerr/10;
file_path = f'../../../../results-correctdatastruct/results{i}.txt'
with open(file_path, 'w') as file:
    file.write(f"tau is {tvals[i]}\n")
    file.write(f"error is {avgerr}\n")
file_path = f'../../../../results-correctdatastruct/trainhist{i}.pkl'
with open(file_path, 'wb') as fp:
    pickle.dump(hist, fp)
