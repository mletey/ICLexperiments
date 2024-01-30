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
from train import train, create_train_state
from model.knn import KnnConfig
from model.mlp import MlpConfig
from model.poly import PolyConfig
from model.transformer import TransformerConfig
from task.regression import LinearRegression
from task.regression import LinearRegressionCorrect

d=10;
tau=0.1
P = ind(d**2*tau);
alpha = 1; N = int(alpha*d);

sigma = 0.25;
psi = 1;

i = int(sys.argv[1]) - 1; # grab value of $SLURM_ARRAY_TASK_ID to index over taus
linobject = LinearRegressionCorrect(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, batch_size = P, seed=None);
config = TransformerConfig(pos_emb=False, n_hidden=512)

l2s = np.array([0,1,0.5,0.1,0.01])
l2 = l2s[i]
state, hist = train(config, data_iter=iter(linobject), loss='mse', test_every=1000, train_iters=1000000, l2_weight = l2, lr=1e-4)

avgerr = 0;
loss_func = optax.squared_error
for _ in range(10):
  xs, labels = next(linobject); # generates data
  logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
  avgerr = avgerr + loss_func(logits, labels).mean()

avgerr = avgerr/10;
file_path = f'../../../../resultsregsweep/results10d{i}.txt'
with open(file_path, 'w') as file:
    file.write(f"tau is {tvals[i]}\n")
    file.write(f"error is {avgerr}\n")
file_path = f'../../../../resultsregsweep/trainhist10d{i}.pkl'
with open(file_path, 'wb') as fp:
    pickle.dump(hist, fp)
