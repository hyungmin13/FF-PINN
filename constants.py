#%%
import numpy as np
import optax
import domain, trackdata, network, problem, equation
import os
import shutil
import pickle
from pathlib import Path
from soap_jax import soap
class ConstantsBase:
    def __getitem__(self, key):
        if key not in vars(self): raise KeyError(f'key "{key}" not defined in class')
        return getattr(self, key)
    def __setitem__(self, key, item):
        if key not in vars(self): raise KeyError(f'key "{key}" not defined in class')
        setattr(self, key, item)
    def __str__(self):
        s = repr(self) + '\n'
        for k in vars(self): s+=f"{k}: {self[k]}\n"
        return s
    
    @property
    def summary_out_dir(self):
        cur_dir = os.getcwd()
        return f"{os.path.dirname(cur_dir)}/{self.run}/summary/"
    @property
    def model_out_dir(self):
        cur_dir = os.getcwd()
        return f"{os.path.dirname(cur_dir)}/{self.run}/models/"
    @property
    def report_out_dir(self):
        cur_dir = os.getcwd()
        return f"{os.path.dirname(cur_dir)}/{self.run}/reports/"
    
    def get_outdirs(self):
        if os.path.exists(self.summary_out_dir):
            print('Loading saved checkpoints')
        else:
            Path(self.summary_out_dir).mkdir(exist_ok=True, parents=True)
        if os.path.exists(self.model_out_dir):
            print('Loading saved checkpoints')
        else:
            Path(self.model_out_dir).mkdir(exist_ok=True, parents=True)
        if os.path.exists(self.report_out_dir):
            print('Loading saved checkpoints')
        else:
            Path(self.report_out_dir).mkdir(exist_ok=True, parents=True)

    def save_constants_file(self):
        if os.path.exists(os.path.dirname(self.summary_out_dir)):
            pass
        else:
            print(os.path.dirname(self.summary_out_dir))
            os.mkdir(os.path.dirname(self.summary_out_dir))
        with open(self.summary_out_dir + f"constants.txt", 'w') as f:
            for k in vars(self): f.write(f"{k}: {self[k]}\n")
        with open(self.summary_out_dir + f"constants.pickle", 'wb') as f:
            pickle.dump(vars(self), f)
        with open(self.report_out_dir + f"reports.txt", 'w') as f:
            f.write(f"{'Steps':{12}} {'Loss':{12}} {'U_loss':{12}} {'V_loss':{12}} {'W_loss':{12}} {'Fu_loss':{12}} {'Fv_loss':{12}} {'Fw_loss':{12}} {'Fp_loss':{12}} {'Con_loss':{12}} {'NS1_loss':{12}} {'NS2_loss':{12}} {'NS3_loss':{12}} {'ENR_loss':{12}} {'U_error':{12}} {'V_error':{12}} {'W_error':{12}} {'T_error':{12}}\n")

    @property
    def constants_file(self):
        return self.summary_out_dir + f"constants.pickle"

def print_c_dicts(c_dicts):
    keys = []
    for c_dict in c_dicts[::-1]:
        for k in c_dict.keys():
            if k not in keys: keys.append(k)

    for k in keys:
        print(f"{k}: ",end="")
        for i,c_dict in enumerate(c_dicts):
            if k in c_dict.keys(): item=str(c_dict[k])
            else: item='None'
            if i == len(c_dicts)-1: print(f"{item}",end="")
            else: print(f"{item} | ",end="")
        print("")    

class Constants(ConstantsBase):
    def __init__(self, **kwargs):
        self.run = "HIT"

        
        self.domain_init_kwargs = dict()
        self.data_init_kwargs = dict()
        self.network_init_kwargs = dict()
        self.problem_init_kwargs = dict()
        self.optimization_init_kwargs = dict()
        self.equation_init_kwargs = dict()
        for key in kwargs.keys(): self[key] = kwargs[key]

        self.domain = domain.Domain
        self.data = trackdata.Data
        self.network = eval('network.'+ self.network_init_kwargs['network_name'])
        self.problem = problem.Problem
        self.equation1 = eval('equation.'+ self.equation_init_kwargs['equation1'])
        self.equation2 = eval('equation.'+ self.equation_init_kwargs['equation2'])
        if self.optimization_init_kwargs['optimiser'] == 'soap':
            self.optimization_init_kwargs['optimiser'] = soap
        else:
            self.optimization_init_kwargs['optimiser'] = optax.adam
        

if __name__ == "__main__":
    from domain import *
    from trackdata import *
    from network import *

    run = "run00"
    all_params = {"domain":{}, "data":{}, "network":{}}

    # Set Domain params
    domain_range = {'t':(0,8), 'x':(0,1.2), 'y':(0,1.2), 'z':(0,1)}
    grid_size = [9, 200, 200, 200]
    bound_keys = ['ic', 'bcxu', 'bcxl', 'bcyu', 'bcyl', 'bczu', 'bczl']
    

    # Set Data params
    path = '/scratch/hyun/UrbanRescue/run065/'
    data_keys = ['pos', 'vel', 'T', ]
    viscosity = 15*10**(-6)
    
    # Set network params
    key = random.PRNGKey(1)
    layer_sizes = [4, 16, 32, 16, 4]
    network_name = 'MLP'



    # Set optimization params
    n_steps = 100000
    optimiser = optax.adam
    learning_rate = 1e-3
    p_batch = 5000
    e_batch = 5000
    b_batch = 5000


    c = Constants(
        run= run,
        domain_init_kwargs = dict(domain_range = domain_range, grid_size = grid_size, bound_keys=bound_keys),
        data_init_kwargs = dict(path = path, viscosity = viscosity, data_keys = data_keys),
        network_init_kwargs = dict(key = key, layer_sizes = layer_sizes, network_name = network_name),
        optimization_init_kwargs = dict(optimiser = optimiser, learning_rate = learning_rate, n_steps = n_steps,
                                        p_batch = p_batch, e_batch = e_batch, b_batch = b_batch)
    )

    c.get_outdirs()
    c.save_constants_file()

# %%
