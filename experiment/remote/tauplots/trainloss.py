import pickle
import sys
import jax
from flax.serialization import from_state_dict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import optax

sys.path.append('../../')
sys.path.append('../../../')
from common import *
from train import train, create_train_state

varsar = []
for i in range(1):
    file_path = f'../../../../resultssgd/longtrainhist10d0.pkl'
    with open(file_path, 'rb') as fp:
        loaded = pickle.load(fp)
    loss_values = [Metrics.loss for Metrics in loaded['train']]
    vals = []
    for loss_array in loss_values:
        vals.append(loss_array.item())
    if i == 0:
        print(vals)
    vals = np.array(vals)
    varsar.append(np.var(vals[650:-1]))

#print(varsar)
