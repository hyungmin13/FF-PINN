#%%
import jax
import jax.numpy as jnp
from jax import random
from time import time
import optax
from jax import value_and_grad
from functools import partial
from jax import jit
from tqdm import tqdm
from typing import Any
from flax import struct
from flax.serialization import to_state_dict, from_state_dict
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.interpolate import interpn
import h5py
from scipy.io import loadmat
import argparse
from Tecplot_mesh import tecplot_Mesh
from tqdm import tqdm
#%%
class Model(struct.PyTreeNode):
    params: Any
    forward: callable = struct.field(pytree_node=False)
    def __apply__(self,*args):
        return self.forward(*args)

class PINNbase:
    def __init__(self,c):
        c.get_outdirs()
        self.c=c

class PINN(PINNbase):
    def test(self):
        all_params = {"domain":{}, "data":{}, "network":{}, "problem":{}}
        all_params["domain"] = self.c.domain.init_params(**self.c.domain_init_kwargs)
        all_params["data"] = self.c.data.init_params(**self.c.data_init_kwargs)
        global_key = random.PRNGKey(42)
        all_params["network"] = self.c.network.init_params(**self.c.network_init_kwargs)
        all_params["problem"] = self.c.problem.init_params(**self.c.problem_init_kwargs)
        optimiser = self.c.optimization_init_kwargs["optimiser"](self.c.optimization_init_kwargs["learning_rate"])
        grids, all_params = self.c.domain.sampler(all_params)
        train_data, all_params = self.c.data.train_data(all_params)
        #all_params = self.c.problem.constraints(all_params)
        #valid_data = self.c.problem.exact_solution(all_params)
        model_fn = c.network.network_fn
        return all_params, model_fn, train_data
    
def equ_func(all_params, g_batch, cotangent, model_fns):
    def u_t(batch):
        return model_fns(all_params, batch)
    def u_tt(batch):
        return jax.jvp(u_t,(batch,), (cotangent, ))[1]
    out_x, out_xx = jax.jvp(u_tt, (g_batch,), (cotangent,))
    return out_x, out_xx

def equ_func2(all_params, g_batch, cotangent, model_fns):
    def u_t(batch):
        return model_fns(all_params, batch)
    out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
    return out, out_t

def Derivatives(dynamic_params, all_params, g_batch, model_fns):
    keys = ['u_ref', 'v_ref', 'w_ref', 'u_ref']

    all_params["network"]["layers"] = dynamic_params
    out, out_x = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out, out_y = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out, out_z = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)    
    uvwp = np.concatenate([out[:,k:(k+1)]*all_params["data"][keys[k]] for k in range(len(keys))],1)
    uvwp[:,-1] = 1.185*uvwp[:,-1]
    uxs = np.concatenate([out_x[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,1] for k in range(len(keys))],1)
    uys = np.concatenate([out_y[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,2] for k in range(len(keys))],1)
    uzs = np.concatenate([out_z[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,3] for k in range(len(keys))],1)
    deriv_mat = np.concatenate([np.expand_dims(uxs,2),np.expand_dims(uys,2),np.expand_dims(uzs,2)],2)
    vor_mag = np.sqrt((deriv_mat[:,1,2]-deriv_mat[:,2,1])**2+
                      (deriv_mat[:,2,0]-deriv_mat[:,0,2])**2+
                      (deriv_mat[:,0,1]-deriv_mat[:,1,0])**2)
    Q = 0.5 * sum(-np.abs(0.5 * (deriv_mat[:, i, j] + deriv_mat[:, j, i]))**2 +
                  np.abs(0.5 * (deriv_mat[:, i, j] - deriv_mat[:, j, i]))**2 
                  for i in range(3) for j in range(3))
    return uvwp, vor_mag, Q, deriv_mat

def Tecplotfile_gen(path, name, all_params, domain_range, output_shape, order, timestep, is_ground, is_mean, model_fn):
    
    # Load the parameters
    pos_ref = all_params["domain"]["in_max"].flatten()
    dynamic_params = all_params["network"].pop("layers")

    # Create the evaluation grid
    gridbase = [np.linspace(domain_range[key][0], domain_range[key][1], output_shape[i]) for i, key in enumerate(['t', 'x', 'y', 'z'])]
    gridbase_n = [gridbase[i].copy()/pos_ref[i] for i in range(len(gridbase))]
    if order[0] == 0:
        if order[1] == 1:
            z_e, y_e, x_e = np.meshgrid(gridbase[-1], gridbase[-2], gridbase[-3], indexing='ij')
            z_n, y_n, x_n = np.meshgrid(gridbase_n[-1], gridbase_n[-2], gridbase_n[-3], indexing='ij')
        else:
            y_e, z_e, x_e = np.meshgrid(gridbase[-2], gridbase[-1], gridbase[-3], indexing='ij')
            y_n, z_n, x_n = np.meshgrid(gridbase_n[-2], gridbase_n[-1], gridbase_n[-3], indexing='ij')
    elif order[0] == 1:
        if order[1] == 0:
            z_e, x_e, y_e = np.meshgrid(gridbase[-1], gridbase[-3], gridbase[-2], indexing='ij')
            z_n, x_n, y_n = np.meshgrid(gridbase_n[-1], gridbase_n[-3], gridbase_n[-2], indexing='ij')
        else:
            y_e, x_e, z_e = np.meshgrid(gridbase[-2], gridbase[-3], gridbase[-1], indexing='ij')
            y_n, x_n, z_n = np.meshgrid(gridbase_n[-2], gridbase_n[-3], gridbase_n[-1], indexing='ij')
    elif order[0] == 2:
        if order[1] == 0:
            x_e, z_e, y_e = np.meshgrid(gridbase[-3], gridbase[-1], gridbase[-2], indexing='ij')
            x_n, z_n, y_n = np.meshgrid(gridbase_n[-3], gridbase_n[-1], gridbase_n[-2], indexing='ij')
        else:
            x_e, y_e, z_e = np.meshgrid(gridbase[-3], gridbase[-2], gridbase[-1], indexing='ij')
            x_n, y_n, z_n = np.meshgrid(gridbase_n[-3], gridbase_n[-2], gridbase_n[-1], indexing='ij')   
    t_e = np.zeros(output_shape[1:]) + gridbase[0][timestep]
    t_n = np.zeros(output_shape[1:]) + gridbase_n[0][timestep]
    eval_grid = np.concatenate([t_n.reshape(-1,1), x_n.reshape(-1,1), y_n.reshape(-1,1), z_n.reshape(-1,1)], axis=1)
    eval_grid_e = np.concatenate([t_e.reshape(-1,1), x_e.reshape(-1,1), y_e.reshape(-1,1), z_e.reshape(-1,1)], axis=1)
    # Load Ground truth data if is_ground is True
    if is_ground:
        ground_data = np.load(path + 'ground/ts_' + str(timestep).zfill(2) + '.npy')
    if is_mean:
        mean_data = np.load(path + 'mean')

    # Evaluate the derivatives
    uvwp, vor_mag, Q, deriv_mat = zip(*[Derivatives(dynamic_params, all_params, eval_grid[i:i+10000], model_fn)
                                        for i in range(0, eval_grid.shape[0], 10000)])
    
    # Concatenate the results
    uvwp = np.concatenate(uvwp, axis=0)
    vor_mag = np.concatenate(vor_mag, axis=0)
    Q = np.concatenate(Q, axis=0)
    deriv_mat = np.concatenate(deriv_mat, axis=0)
    uvwp[:,3] = uvwp[:,3] - np.mean(uvwp[:,3])

    if is_ground:
        grounds = [ground_data[:,i+4].reshape(output_shape[1:]) for i in range(3)]
        errors = [np.sqrt(np.square(uvwp[:,i].reshape(output_shape[1:]) - grounds[i])) for i in range(3)]
        if ground_data.shape[1] > 7:
            p_ground = ground_data[:,7].reshape(output_shape[1:])
            p_error = np.sqrt(np.square(uvwp[:,3].reshape(output_shape[1:]) - p_ground))
        if ground_data.shape[1] > 8:
            temp_ground = ground_data[:,8].reshape(output_shape[1:])
            temp_error = np.sqrt(np.square(uvwp[:,4].reshape(output_shape[1:]) - temp_ground))
    if is_mean:
        means = [mean_data['vel'][:,i].reshape(output_shape[1:]) for i in range(3)]
        flucs = [uvwp[:,i].reshape(output_shape[1:]) - means[i] for i in range(3)]

    # Tecplot file generation
    filename = path + 'Tecplotfile/' + name + '/ts_' + str(timestep) + '.dat'
    if os.path.isdir(path + 'Tecplotfile/' + name):
        pass
    else:
        os.mkdir(path + 'Tecplotfile/' + name)
    X, Y, Z = output_shape[1:]
    vars = [('u_pred[m/s]',np.float32(uvwp[:,0].reshape(-1))), ('v_pred[m/s]',np.float32(uvwp[:,1].reshape(-1))), 
            ('w_pred[m/s]',np.float32(uvwp[:,2].reshape(-1))), ('p_pred[Pa]',np.float32(uvwp[:,3].reshape(-1))),
            ('Q[1/s^2]', np.float32(Q.reshape(-1)))]
    if is_ground:
        vars += [('u_error[m/s]', np.float32(errors[0].reshape(-1))),
                 ('v_error[m/s]', np.float32(errors[1].reshape(-1))),
                 ('w_error[m/s]', np.float32(errors[2].reshape(-1)))]
        if ground_data.shape[1] > 7:
            vars += [('p_error[Pa]', np.float32(p_error.reshape(-1)))]
        if ground_data.shape[1] > 8:
            vars += [('temp_error[K]', np.float32(temp_error.reshape(-1)))]
    if is_mean:
        vars += [('u_fluc[m/s]', np.float32(flucs[0].reshape(-1))),
                 ('v_fluc[m/s]', np.float32(flucs[1].reshape(-1))),
                 ('w_fluc[m/s]', np.float32(flucs[2].reshape(-1)))]
    pad = 27
    tecplot_Mesh(filename, X, Y, Z, x_e.reshape(-1), y_e.reshape(-1), z_e.reshape(-1), vars, pad)

    if os.path.isdir(path + 'npyresult/' + name):
        pass
    else:
        print('check')
        os.mkdir(path + 'npyresult/' + name)
    np.save(path + 'npyresult/' + name + f'/ts_{timestep:02d}' + '.npy', np.concatenate([eval_grid_e, uvwp, deriv_mat[:,:,0], deriv_mat[:,:,1], deriv_mat[:,:,2]], axis=1))
#%%
if __name__ == "__main__":
    from domain import *
    from trackdata import *
    from network import *
    from constants import *
    from problem import *
    from txt_reader import *
    import os
    parser = argparse.ArgumentParser(description='PINN')
    parser.add_argument('-f', '--foldername', type=str, help='foldername', default='HIT')
    parser.add_argument('-c', '--config', type=str, help='configuration', default='eval_config')
    args = parser.parse_args()

    # Get evaluation configuration
    cur_dir = os.getcwd()
    config_txt = cur_dir + '/' + args.config + '.txt'
    data = parse_tree_structured_txt(config_txt)

    # Get model constants
    with open(os.path.dirname(cur_dir)+ '/' + data['path'] + args.foldername +'/summary/constants.pickle','rb') as f:
        constants = pickle.load(f)
    values = list(constants.values())

    c = Constants(run = values[0],
                domain_init_kwargs = values[1],
                data_init_kwargs = values[2],
                network_init_kwargs = values[3],
                problem_init_kwargs = values[4],
                optimization_init_kwargs = values[5],
                equation_init_kwargs = values[6],)
    run = PINN(c)

    # Get model parameters
    checkpoint_list = sorted(glob(run.c.model_out_dir+'/*.pkl'), key=lambda x: int(x.split('_')[-1].split('.')[0]))

    with open(checkpoint_list[-1],"rb") as f:
        model_params = pickle.load(f)
    all_params, model_fn, train_data = run.test()
    model = Model(all_params["network"]["layers"], model_fn)
    all_params["network"]["layers"] = from_state_dict(model, model_params).params
    domain_range = data['tecplot_init_kwargs']['domain_range']
    output_shape = data['tecplot_init_kwargs']['out_shape']
    order = data['tecplot_init_kwargs']['order']
    timesteps = data['tecplot_init_kwargs']['timestep']
    is_ground = data['tecplot_init_kwargs']['is_ground']
    path = data['tecplot_init_kwargs']['path']
    is_mean = data['tecplot_init_kwargs']['is_mean']
    path = os.path.dirname(cur_dir) + '/' + path
    pos_ref = all_params["domain"]["in_max"].flatten()
    for timestep in timesteps:
        Tecplotfile_gen(path, args.foldername, all_params, domain_range, output_shape, order, timestep, is_ground, is_mean, model_fn)