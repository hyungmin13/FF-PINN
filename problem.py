#%%
import jax.nn
import jax.numpy as jnp
import numpy as np
import h5py
from glob import glob
import os
from trackdata import *
class Problembase:
    @staticmethod
    def init_params(*args):
        raise NotImplementedError
    @staticmethod
    def exact_solution(all_params):
        raise NotImplementedError

class Problem(Problembase):
    @staticmethod
    def init_params(**kwargs):
        problem_params = {}
        for key, value in kwargs.items():
            problem_params[key] = value
        return problem_params
    @staticmethod
    def exact_solution(all_params):
        path_s = all_params["problem"]["path_s"]
        data_keys = all_params["data"]["data_keys"]
        viscosity = all_params["data"]["viscosity"]
        u_ref = all_params["data"]["u_ref"]
        all_params["data"] = Data.init_params(path = path_s, data_keys = data_keys, 
                                              viscosity = viscosity)

        valid_data, _ = Data.train_data(all_params)
        return valid_data
    
if __name__ == "__main__":
    from trackdata import *
    from domain import *
    all_params = {"problem":{}, "data":{}, "domain":{}}

    domain_range = {'t':(0,7.4), 'x':(0,8), 'y':(0,8), 'z':(0,1)}
    grid_size = [51, 200, 200, 200]
    bound_keys = ['ic', 'bcxu', 'bcxl', 'bcyu', 'bcyl', 'bczu', 'bczl']

    path = '/RBC_G8_DNS/npdata/lv6_xbound/'
    data_keys = ['pos', 'vel', 'T']
    viscosity = 15*10e-6

    loss_weights = (1,1,1,0.00001,0.00001,0.00001,0.00001)
    path_s = '/RBC_G8_DNS/npdata/lv6_xbound/'
    all_params["data"] = Data.init_params(path = path, data_keys = data_keys, viscosity = viscosity)
    all_params["domain"] = Domain.init_params(domain_range = domain_range, bound_keys = bound_keys, grid_size = grid_size)
    all_params["problem"] = Problem.init_params(loss_weights = loss_weights, path_s = path_s)
    datas = Problem.exact_solution(all_params)