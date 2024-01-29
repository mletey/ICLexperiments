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

for i in range(14):
    file_path = f'../../../../results-correctdatastruct/results5d{i}.txt'
    with open(file_path, 'r') as file:
        file_contents = file.read()
    print(file_contents)

