import jax
from flax.serialization import from_state_dict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import optax
import pickle
import sys
sys.path.append('../../')
sys.path.append('../../../')
from common import *
from traintheory import train, create_train_state
from model.knn import KnnConfig
from model.mlp import MlpConfig
from model.poly import PolyConfig
from model.transformer import TransformerConfig
from task.regression import LinearRegression
from task.regression import LinearRegressionCorrect

d=10;
tvals = [0.1,1,10,100];
alpha = 1; N = int(alpha*d);
hiddens = [10,20,50,100]

sigma = 0.25;
psi = 1;

myname = sys.argv[1] # grab value of $mydir to add results
i = int(sys.argv[2]) - 1; # grab value of $SLURM_ARRAY_TASK_ID to index over taus + model sizes
a = i % len(tvals); 
P = int(tvals[a]*(d**2));
b = int(i/len(tvals)); 
h = hiddens[b];

linobject = LinearRegressionCorrect(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, batch_size = P, seed=None);
config = TransformerConfig(pos_emb=False, n_hidden=h)

state, hist = train(config, data_iter=iter(linobject), loss='mse', test_every=1000, train_iters=500000, optim=optax.adamw,lr=1e-4)

avgerr = 0;
loss_func = optax.squared_error
numsamples = 10000
for _ in range(numsamples):
  xs, labels = next(linobject); # generates data
  logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
  avgerr = avgerr + loss_func(logits, labels).mean()
avgerr = avgerr/numsamples;

file_path = f'./{myname}/error-P{P}-h{h}.txt'
with open(file_path, 'w') as file:
    file.write(f'{avgerr}')
file_path = f'./{myname}/train-P{P}-h{h}.pkl'
with open(file_path, 'wb') as fp:
    pickle.dump(hist, fp)
