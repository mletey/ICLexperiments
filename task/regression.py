"""
In-context regression tasks

author: William Tong (wtong@g.harvard.edu)
modified by Mary Letey for 2024 In-Context Learning Project
"""

# <codecell>

import numpy as np

class LinearRegression:
    def __init__(self, n_points=6, n_dims=2, eta_scale=1, w_scale=1, batch_size=128, seed=None) -> None:
        self.n_points = n_points # n_points = N+1 where N = context length, as n_points includes the (N+1)st query vector
        self.n_dims = n_dims # d = dimension of tokens
        self.w_scale = w_scale # sigma_beta
        self.eta_scale = eta_scale # noise sigma
        self.batch_size = batch_size # P = number of contexts
        self.rng = np.random.default_rng(seed)
    
    def __next__(self):
        xs = self.rng.normal(loc=0, scale = 1/np.sqrt(self.n_points), size=(self.batch_size, self.n_points, self.n_dims))
        ws = self.rng.normal(loc=0, scale = self.w_scale, size=(self.batch_size, self.n_dims, 1))
        ys = xs @ ws + self.rng.normal(loc=0, scale = self.eta_scale, size=(self.batch_size, self.n_points, 1))
        zs = np.zeros((self.batch_size, self.n_points, self.n_dims - 1))
        ys_pad = np.concatenate((ys, zs), axis=-1)

        interl_xs = np.empty((self.batch_size, self.n_points * 2 - 1, self.n_dims))
        interl_xs[:, 0::2] = xs
        interl_xs[:, 1::2] = ys_pad[:,:-1]
	# for a given context, the even vectors are xs and the odd vectors are [corresponding y val, 0,0,0,0].  The final vector is the x_(N+1) query vector

	# returns the interl_xs configuration and the true N+1 value for testing 
        return interl_xs, ys[:,-1].squeeze()


    def __iter__(self):
        return self

class LinearRegressionCorrect:
    def __init__(self, n_points=6, n_dims=2, eta_scale=1, w_scale=1, batch_size=128, seed=None) -> None:
        self.n_points = n_points # n_points = N+1 where N = context length, as n_points includes the (N+1)st query vector
        self.n_dims = n_dims # d = dimension of tokens
        self.w_scale = w_scale # sigma_beta
        self.eta_scale = eta_scale # noise sigma
        self.batch_size = batch_size # P = number of contexts
        self.rng = np.random.default_rng(seed)
    
    def __next__(self):
        xs = self.rng.normal(loc=0, scale = 1/np.sqrt(self.n_points), size=(self.batch_size, self.n_points, self.n_dims))
        ws = self.rng.normal(loc=0, scale = self.w_scale, size=(self.batch_size, self.n_dims, 1))
        ys = xs @ ws + self.rng.normal(loc=0, scale = self.eta_scale, size=(self.batch_size, self.n_points, 1))
        Z = np.zeros((self.batch_size, self.n_points, self.n_dims + 1))
        Z[:,:,0:self.n_dims] = xs
        Z[:,:,-1] = ys.squeeze()
        Z[:,-1, self.n_dims] = 0 #padding for final context

        #Z = np.transpose(Z,(0,2,1))
	    
	# returns the Z [x,y,x,y]... configuration and the true N+1 value for testing 
        return Z, ys[:,-1].squeeze()

    def __iter__(self):
        return self

class FiniteLinearRegression:
    """Based on the construction described in Raventos et al. 2023"""

    def __init__(self, n_ws=128, n_points=16, n_dims=8, noise_scale=0.5, batch_size=128, seed=None, reset_rng_for_data=True) -> None:
        self.n_points = n_points
        self.n_dims = n_dims
        self.noise_scale = noise_scale
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        self.ws = None
        if n_ws is not None:
            self.ws = self.rng.standard_normal(size=(n_ws, n_dims))
        
        if reset_rng_for_data:
            self.rng = np.random.default_rng(None)
    
    def __next__(self):
        xs = self.rng.standard_normal(size=(self.batch_size, self.n_points, self.n_dims))
        if self.ws is None:
            ws = self.rng.standard_normal(size=(self.batch_size, self.n_dims))
        else:
            ws_idxs = self.rng.choice(len(self.ws), size=(self.batch_size), replace=True)
            ws = self.ws[ws_idxs]
        
        ws = np.expand_dims(ws, axis=-1)
        ys = xs @ ws + self.rng.normal(scale=self.noise_scale, size=(self.batch_size, self.n_points, 1))
        zs = np.zeros((self.batch_size, self.n_points, self.n_dims - 1))
        ys_pad = np.concatenate((ys, zs), axis=-1)

        interl_xs = np.empty((self.batch_size, self.n_points * 2 - 1, self.n_dims))
        interl_xs[:, 0::2] = xs
        interl_xs[:, 1::2] = ys_pad[:,:-1]
        return interl_xs, ys[:,-1].squeeze()


    def __iter__(self):
        return self

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # task = LinearRegression(batch_size=5, n_dims=1)
    # xs, ys = next(task)

    # plt.scatter(xs[0][0:-1:2], xs[0][1::2])
    # plt.scatter([xs[0][-1]], ys[0])

    # task = LinearRegression(batch_size=5, n_dims=2, n_points=500, seed=1)
    # xs, ys = next(task)

    task = FiniteLinearRegression(n_ws=None, batch_size=5, n_dims=2, n_points=500, seed=1)
    xs, ys = next(task)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(xs[0][0:-1:2, 0], xs[0][0:-1:2, 1], xs[0][1::2, 0], alpha=0.3)
    ax.scatter(xs[0][-1,0], xs[0][-1, 1], ys[0])

