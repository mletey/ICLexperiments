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


i = int(sys.argv[1])
file_path = f'../../../../results-correctdatastruct/trainhist10d{i}.pkl'
with open(file_path, 'rb') as fp:
    loaded = pickle.load(fp)

loss_values = [Metrics.loss for Metrics in loaded['train']]
vals = []
for loss_array in loss_values:
    vals.append(loss_array.item())
print(vals)
