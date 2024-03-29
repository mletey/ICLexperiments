import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import optax

sys.path.append('../../../')
sys.path.append('../../../../')
from common import *
from train import train, create_train_state

mydir = sys.argv[1]
myiteration = sys.argv[2]

trainvals = []
testvals = []

for i in range(100):
    file_path = f'./{mydir}/pickles/train-{i}-{myiteration}.pkl'
    with open(file_path, 'rb') as fp:
        loaded = pickle.load(fp)
    trainloss = [Metrics.loss for Metrics in loaded['train']]
    loss_array = trainloss[-1]
    trainvals.append(loss_array.item())

print(trainvals)
cutoff = 0.001
overparam = [i for i in range(len(trainvals)) if trainvals[i] < cutoff]
print("Tau Inflection at tau = ",overparam[-1])
taus = range(1,101)

def growth(myarr):
    answ = []
    for i in range(len(myarr)):
        if i == 0:
            answ.append(myarr[1]/myarr[0])
        else:
            answ.append(myarr[i]/myarr[i-1])
    return answ


plt.plot(taus[0:30],trainvals[0:30],label='Final Training Error')
#plt.axvline(x=overparam[-1],label='tau inflection')
# deltas = growth(trainvals)
# plt.plot(taus[0:30],deltas[0:30],label='delta Training Error')
plt.axvline(x=25,label='empirical inflection')
plt.title(f'Train Error Regime Transition, Seed = {myiteration}')
plt.legend()
plt.savefig(f'./{mydir}/transition-{myiteration}.png')

def computeslope(myarr):
    answ=[]
    for i in range(len(myarr)):
        if i == 0:
            answ.append(myarr[1]-myarr[0])
        elif i == len(myarr)-1:
            answ.append(myarr[i]-myarr[i-1])
        else:
            answ.append((myarr[i+1]-myarr[i-1])/2)
    return answ


# trainslopes = computeslope(trainvals)
# plt.plot(range(40),trainslopes[0:40],label='slopes approx')
# plt.axvline(x=24,label='approx max')
# plt.legend()
# plt.savefig(f'./{mydir}/testplot-{myiteration}.png')
