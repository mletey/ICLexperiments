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

val=sys.argv[1]
file_path = f'./{val}/resulttest.txt'
with open(file_path, 'w') as file:
    file.write('hello wurld')
