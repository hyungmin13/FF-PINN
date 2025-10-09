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

class Model(struct.PyTreeNode):
    params: Any
    forward: callable = struct.field(pytree_node=False)
    def __apply__(self,*args):
        return self.forward(*args)

class PINNbase:
    def __init__(self,c):
        c.get_outdirs()
        c.save_constants_file()
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
        all_params = self.c.problem.constraints(all_params)
        valid_data = self.c.problem.exact_solution(all_params)
        model_fn = c.network.network_fn
        return all_params, model_fn, train_data, valid_data
    
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
    return uvwp, vor_mag, Q

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
    all_params, model_fn, train_data, valid_data = run.test()
    model = Model(all_params["network"]["layers"], model_fn)
    all_params["network"]["layers"] = from_state_dict(model, model_params).params
    if data['temporal_init_kwargs']['is_error'] == 1:
        print('HEEEEEEEEEEEEEEEEY')
    if data['behavior_init_kwargs']['is_error'] == 1:
        print('HEEEEEEEEEEEEEEEEY')
    if data['tecplot_init_kwargs']['is_error'] == 1:
        print('HEEEEEEEEEEEEEEEY')


    print(checkpoint_list)
    print(data.keys())
    print(constants)
    """

    # Get model parameters
    checkpoint_list = sorted(glob(run.c.model_out_dir+'/*.pkl'), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(checkpoint_list)
    with open(checkpoint_list[-1],"rb") as f:
        model_params = pickle.load(f)
    
    #checkpoint_list = glob(os.path.dirname(cur_dir)+checkpoint_fol+'/models/*.pkl')
    #savefiles = sorted([checkpoint_list[i].split('/')[-1] for i in range(len(checkpoint_list))], key=lambda x: int(x.split('_')[]))
    all_params, model_fn, train_data, valid_data = run.test()

    model = Model(all_params["network"]["layers"], model_fn)
    all_params["network"]["layers"] = from_state_dict(model, model_params).params
    
#%%
    output_shape = (129,129,129)
#%%
    timestep = 25
    pos_ref = all_params["domain"]["in_max"].flatten()
    vel_ref = np.array([all_params["data"]["u_ref"],
                        all_params["data"]["v_ref"],
                        all_params["data"]["w_ref"]])
#%%
    eval_grid = valid_data['pos'].reshape(51,129,129,129,4)[timestep,:,:,:,1:].reshape(-1,3)
    eval_grid = np.concatenate([eval_grid[:,0:1]*pos_ref[1], eval_grid[:,1:2]*pos_ref[2], eval_grid[:,2:3]*pos_ref[3]],1).reshape(output_shape+(3,))
    eval_grid_n = valid_data['pos'].reshape(51,129,129,129,4)[timestep,:,:,:,:].reshape(-1,4)
#%%
    eval_grid_z = eval_grid.reshape(output_shape+(3,))
    x_e = eval_grid_z[:,:,:,1]
    y_e = eval_grid_z[:,:,:,0]
    z_e = eval_grid_z[:,:,:,2]

#%%
    dynamic_params = all_params["network"].pop("layers")
    uvwp, vor_mag, Q = zip(*[Derivatives(dynamic_params, all_params, eval_grid_n[i:i+10000], model_fn) 
                             for i in tqdm(range(0, eval_grid_n.shape[0], 10000),desc="Derivatives")])
    uvwp = np.concatenate(uvwp, axis=0)
    vor_mag = np.concatenate(vor_mag, axis=0)
    Q = np.concatenate(Q, axis=0)

    p_cent = uvwp[:,3].reshape(output_shape) - np.mean(uvwp[:,3].reshape(output_shape))

    u_error = np.sqrt(np.square(uvwp[:,0].reshape(output_shape) - valid_data['vel'].reshape((51,)+output_shape+(4,))[timestep,:,:,:,0].reshape(output_shape)))
    v_error = np.sqrt(np.square(uvwp[:,1].reshape(output_shape) - valid_data['vel'].reshape((51,)+output_shape+(4,))[timestep,:,:,:,1].reshape(output_shape)))
    w_error = np.sqrt(np.square(uvwp[:,2].reshape(output_shape) - valid_data['vel'].reshape((51,)+output_shape+(4,))[timestep,:,:,:,2].reshape(output_shape)))
    p_error = np.sqrt(np.square(uvwp[:,3].reshape(output_shape) - valid_data['vel'].reshape((51,)+output_shape+(4,))[timestep,:,:,:,3].reshape(output_shape)))

#%%
    filename = "datas/"+checkpoint_fol+"/HIT_eval_"+str(timestep)+".dat"
    if os.path.isdir("datas/"+checkpoint_fol):
        pass
    else:
        os.mkdir("datas/"+checkpoint_fol)
    X, Y, Z = (y_e[0,0,:].shape[0], x_e[0,:,0].shape[0], z_e[:,0,0].shape[0])
    vars = [('u_ground[m/s]', np.float32(valid_data['vel'].reshape((51,)+output_shape+(4,))[timestep,:,:,:,0].reshape(-1))),
            ('v_ground[m/s]', np.float32(valid_data['vel'].reshape((51,)+output_shape+(4,))[timestep,:,:,:,1].reshape(-1))), 
            ('w_ground[m/s]', np.float32(valid_data['vel'].reshape((51,)+output_shape+(4,))[timestep,:,:,:,2].reshape(-1))), 
            ('p_ground[Pa]', np.float32(valid_data['vel'].reshape((51,)+output_shape+(4,))[timestep,:,:,:,3].reshape(-1))),
            ('u_pred[m/s]',np.float32(uvwp[:,0].reshape(-1))), ('v_pred[m/s]',uvwp[:,1].reshape(-1)),
            ('w_pred[m/s]',uvwp[:,2].reshape(-1)), ('p_pred[Pa]',uvwp[:,3].reshape(-1)),
            ('u_error[m/s]',u_error.reshape(-1)), ('v_error[m/s]',v_error.reshape(-1)), ('w_error[m/s]',w_error.reshape(-1)), ('p_error[Pa]',p_error.reshape(-1)),
            ('vormag[1/s]',vor_mag.reshape(-1)), ('Q[1/s^2]', Q.reshape(-1))]
    fw = 27
    tecplot_Mesh(filename, X, Y, Z, y_e.reshape(-1), x_e.reshape(-1), z_e.reshape(-1), vars, fw)

#%%
    import os
    cur_dir = os.getcwd()
    test_string = cur_dir + 'saved_dic_58000.pkl'
# %%
    print(test_string.split('/'))    
# %%
    """