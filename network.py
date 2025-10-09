#%%
import jax.numpy as jnp
from jax import random
import numpy as np
class Network:
    @staticmethod
    def init_params(*args):
        raise NotImplementedError

class MLP(Network):
    def __init__(self, all_params):
        self.all_params = all_params
        

    @staticmethod
    def init_params(key, layer_sizes, network_name):
        key_network = random.PRNGKey(key)
        keys = random.split(key_network, len(layer_sizes)-1)
        params = [eval(network_name)._random_layer_params(k, m, n)
                for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]
        network_params = {"layers": params, "layer_sizes":layer_sizes, "network_name":network_name}
        return network_params
    
    @staticmethod
    def _random_layer_params(key, m, n):
        "Create a random layer parameters"

        w_key, b_key = random.split(key)
        w = random.normal(w_key, (m, n))
        b = jnp.zeros((1,n))
        g = jnp.ones((1,n))
        return w,b,g

    @staticmethod
    def network_fn(all_params, x):
        params = all_params["network"]["layers"]
        inmin = all_params["domain"]["in_min"]
        inmax = all_params["domain"]["in_max"]
        x = 2*(x - inmin)/(inmax-inmin) - 1
        for w, b, g in params[:-1]:
            x = g*jnp.dot(x,w/jnp.linalg.norm(w,axis=0, keepdims=True)) + b
            x = jnp.tanh(x)
        w, b, g = params[-1]
        x = jnp.dot(x,w/jnp.linalg.norm(w,axis=0, keepdims=True)) + b
        return x

    def network_fn3(self, x):
        params = self.all_params["network"]["layers"]
        inmin = self.all_params["domain"]["in_min"]
        inmax = self.all_params["domain"]["in_max"]
        x = 2*(x - inmin)/(inmax-inmin) - 1
        for w, b, g in params[:-1]:
            x = g*jnp.dot(x,w/jnp.linalg.norm(w,axis=0, keepdims=True)) + b
            x = jnp.tanh(x)
        w, b, g = params[-1]
        x = jnp.dot(x,w/jnp.linalg.norm(w,axis=0, keepdims=True)) + b
        return x


if __name__=="__main__":
    from domain import *

    all_params = {"network":{}, "domain":{}}
    domain_range = {'t':(0,8), 'x':(0,1.2), 'y':(0,1.2), 'z':(0,1)}
    frequency = 3000
    grid_size = [9, 200, 200, 200]
    bound_keys = ['ic', 'bcxu', 'bcxl', 'bcyu', 'bcyl', 'bczu', 'bczl']
    all_params["domain"] = Domain.init_params(domain_range, frequency, grid_size, bound_keys)
    grids, all_params = Domain.sampler(all_params)
    x = jnp.ones((100,4))
    key = random.PRNGKey(0)
    layer_sizes = [4,16,32,16,1]
    network = MLP
    all_params["network"] = network.init_params(key, layer_sizes)

