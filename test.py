from utils import N4SID_simple
import numpy as np 

def generate_observations(A, x0, n_obs, dt):
    obs = np.zeros((n_obs, *x0.shape))
    obs[0] = x0
    for i in range(1, n_obs):
        obs[i] = A @ obs[i-1]

    return obs 

A = np.array([
    [2, 0],
    [0, 1]
])
x0 = np.array([1, 0])
n_obs = 5
dt = 1
obs = generate_observations(A, x0, n_obs, dt)  

print(obs)
