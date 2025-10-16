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
from soap_jax import soap
import tqdm
import itertools
class Model(struct.PyTreeNode):
    params: Any
    forward: callable = struct.field(pytree_node=False)
    def __apply__(self,*args):
        return self.forward(*args)

@partial(jax.jit, static_argnums=(1, 2, 5, 12))
def PINN_update1(model_states, optimiser_fn, equation_fn, dynamic_params, static_params, static_keys, grids, ff_grid, ff_val, particles, particle_vel, particle_bd, model_fn):
    static_leaves, treedef = static_keys
    leaves = [d if s is None else s for d, s in zip(static_params, static_leaves)]
    all_params = jax.tree_util.tree_unflatten(treedef, leaves)
    lossval, grads = value_and_grad(equation_fn, argnums=0)(dynamic_params, all_params, ff_grid, ff_val, particles, particle_vel, particle_bd, model_fn)
    updates, model_states = optimiser_fn(grads, model_states, dynamic_params)
    dynamic_params = optax.apply_updates(dynamic_params, updates)
    return lossval, model_states, dynamic_params

@partial(jax.jit, static_argnums=(1, 2, 5, 13))
def PINN_update2(model_states, optimiser_fn, equation_fn, dynamic_params, static_params, static_keys, loss_factor, grids, ff_grid, ff_val, particles, particle_vel, particle_bd, model_fn):
    static_leaves, treedef = static_keys
    leaves = [d if s is None else s for d, s in zip(static_params, static_leaves)]
    all_params = jax.tree_util.tree_unflatten(treedef, leaves)
    lossval, grads = value_and_grad(equation_fn, argnums=0)(dynamic_params, all_params, loss_factor, grids, ff_grid, ff_val, particles, particle_vel, particle_bd, model_fn)
    updates, model_states = optimiser_fn(grads, model_states, dynamic_params)
    dynamic_params = optax.apply_updates(dynamic_params, updates)
    return lossval, model_states, dynamic_params

@partial(jax.jit, static_argnums=())
def jit_sample(key, data, shape: int):
    N = data.shape[0]
    idx = random.randint(key, (shape,), 0, N)
    return jnp.take(data, idx, axis=0)

class PINNbase:
    def __init__(self,c):
        c.get_outdirs()
        c.save_constants_file()
        self.c=c

class PINN(PINNbase):
    def train(self):
        all_params = {"domain":{}, "data":{}, "network":{}, "problem":{}}
        all_params["domain"] = self.c.domain.init_params(**self.c.domain_init_kwargs)
        all_params["data"] = self.c.data.init_params(**self.c.data_init_kwargs)
        global_key = random.PRNGKey(42)
        key, network_key = random.split(global_key)
        all_params["network"] = self.c.network.init_params(**self.c.network_init_kwargs)
        all_params["problem"] = self.c.problem.init_params(**self.c.problem_init_kwargs)

        # Initialize optmiser
        learn_rate = optax.exponential_decay(self.c.optimization_init_kwargs["learning_rate"],
                                             self.c.optimization_init_kwargs["decay_step"],
                                             self.c.optimization_init_kwargs["decay_rate"],)
        optimiser = self.c.optimization_init_kwargs["optimiser"](learning_rate=learn_rate, b1=0.95, b2=0.95,
                                                                 weight_decay=0.01, precondition_frequency=5)
        model_states = optimiser.init(all_params["network"]["layers"])
        optimiser_fn = optimiser.update
        model_fn = c.network.network_fn
        dynamic_params = all_params["network"].pop("layers")

        # Define equation function
        equation_fn1 = self.c.equation1.Loss
        report_fn1 = self.c.equation1.Loss_report
        equation_fn2 = self.c.equation2.Loss
        report_fn2 = self.c.equation2.Loss_report
        # Input data and grids
        grids, all_params = self.c.domain.sampler(all_params)
        train_data1, all_params = self.c.data.train_data(all_params)
        train_data2 = self.c.data.ff_data(all_params.copy())
        train_data2_ = np.concatenate([train_data2['vel'], train_data2['p'].reshape(-1,1)],1)
        valid_data = self.c.problem.exact_solution(all_params.copy())

        # Input key initialization
        key, batch_key = random.split(key)
        num_keysplit = 10
        keys = random.split(batch_key, num = num_keysplit)
        keys_split = [random.split(keys[i], num = self.c.optimization_init_kwargs["n_steps1"] + self.c.optimization_init_kwargs["n_steps2"]) for i in range(num_keysplit)]
        keys_iter = [iter(keys_split[i]) for i in range(num_keysplit)]
        keys_next = [next(keys_iter[i]) for i in range(num_keysplit)]

        # Static parameters
        leaves, treedef = jax.tree_util.tree_flatten(all_params)
        static_params = tuple(x if isinstance(x,(np.ndarray, jnp.ndarray)) else None for x in leaves)
        static_leaves = tuple(None if isinstance(x,(np.ndarray, jnp.ndarray)) else x for x in leaves)
        static_keys = (static_leaves, treedef)

        # Initializing batches

        #idx = jax.random.randint(keys_next[0], (10000,), 0, 5242081)
        #p_batch = jnp.take(train_data1['pos'], idx, axis=0)
        #v_batch = jnp.take(train_data1['vel'], idx, axis=0)
        N1 = train_data1['pos'].shape[0]
        N2 = train_data2['pos'].shape[0]
        B = 10000
        perm1 = random.permutation(keys_next[0], N1)
        perm2 = random.permutation(keys_next[0], N2)
        data_p = []
        data_v = []
        data_fp = []
        data_fv = []
        for i in range(N1//B):
            batch_p = train_data1['pos'][perm1[i*B:(i+1)*B],:]
            batch_v = train_data1['vel'][perm1[i*B:(i+1)*B],:]
            data_p.append(batch_p)
            data_v.append(batch_v)
        data_p.append(train_data1['pos'][perm1[-10001:-1],:])
        data_v.append(train_data1['vel'][perm1[-10001:-1],:])
        for i in range(N2//B):
            batch_fp = train_data2['pos'][perm2[i*B:(i+1)*B],:]
            batch_fv = train_data2_ [perm2[i*B:(i+1)*B],:]
            data_fp.append(batch_fp)
            data_fv.append(batch_fv)
        data_fp.append(train_data2['pos'][perm2[-10001:-1],:])
        data_fv.append(train_data2_ [perm2[-10001:-1],:])
        p_batches = itertools.cycle(data_p)
        v_batches = itertools.cycle(data_v)
        fp_batches = itertools.cycle(data_fp)
        fv_batches = itertools.cycle(data_fv)
        p_batch = next(p_batches)
        v_batch = next(v_batches)
        ffgrid_batch = next(fp_batches)
        ffval_batch = next(fv_batches)

        g_batch = jnp.stack([random.choice(keys_next[k+1], 
                                           grids['eqns'][arg], 
                                           shape=(self.c.optimization_init_kwargs["e_batch"],)) 
                             for k, arg in enumerate(list(all_params["domain"]["domain_range"].keys()))],axis=1)
        b_batches = []
        for b_key in all_params["domain"]["bound_keys"]:
            b_batch = jnp.stack([random.choice(keys_next[k+5], 
                                            grids[b_key][arg], 
                                            shape=(self.c.optimization_init_kwargs["e_batch"],)) 
                                for k, arg in enumerate(list(all_params["domain"]["domain_range"].keys()))],axis=1)
            b_batches.append(b_batch)

        # Initializing the update function
        update = PINN_update1.lower(model_states, optimiser_fn, equation_fn1, dynamic_params, static_params, static_keys, g_batch, ffgrid_batch, ffval_batch, p_batch, v_batch, b_batches, model_fn).compile()
        
        # Training loop
        for i in range(self.c.optimization_init_kwargs["n_steps1"]):
            keys_next = [next(keys_iter[i]) for i in range(num_keysplit)]
            p_batch = next(p_batches)
            v_batch = next(v_batches)
            ffgrid_batch = next(fp_batches)
            ffval_batch = next(fv_batches)
            g_batch = jnp.stack([random.choice(keys_next[k+1], 
                                            grids['eqns'][arg], 
                                            shape=(self.c.optimization_init_kwargs["e_batch"],)) 
                                for k, arg in enumerate(list(all_params["domain"]["domain_range"].keys()))],axis=1)
            b_batches = []
            for b_key in all_params["domain"]["bound_keys"]:
                b_batch = jnp.stack([random.choice(keys_next[k+5], 
                                                grids[b_key][arg], 
                                                shape=(self.c.optimization_init_kwargs["e_batch"],)) 
                                    for k, arg in enumerate(list(all_params["domain"]["domain_range"].keys()))],axis=1)
                b_batches.append(b_batch)
            lossval, model_states, dynamic_params = update(model_states, dynamic_params, static_params, g_batch, ffgrid_batch, ffval_batch, p_batch, v_batch, b_batches)
        
        
            self.report1(i, report_fn1, dynamic_params, all_params, p_batch, v_batch, g_batch, ffgrid_batch, ffval_batch, b_batch, valid_data, keys_iter[-1], self.c.optimization_init_kwargs["save_step"], model_fn)
            self.save_model(i, dynamic_params, all_params, self.c.optimization_init_kwargs["save_step"], model_fn)
        loss_factor = jnp.exp(0)
        update = PINN_update2.lower(model_states, optimiser_fn, equation_fn2, dynamic_params, static_params, static_keys, loss_factor, g_batch, ffgrid_batch, ffval_batch, p_batch, v_batch, b_batches, model_fn).compile()
        j = i
        # Training loop
        for k in range(self.c.optimization_init_kwargs["n_steps2"]):
            i = j + k
            keys_next = [next(keys_iter[i]) for i in range(num_keysplit)]
            #p_batch = random.choice(keys_next[0],train_data1['pos'],shape=(self.c.optimization_init_kwargs["p_batch"],))
            #v_batch = random.choice(keys_next[0],train_data1['vel'],shape=(self.c.optimization_init_kwargs["p_batch"],))
            p_batch = next(p_batches)
            v_batch = next(v_batches)
            ffgrid_batch = next(fp_batches)
            ffval_batch = next(fv_batches)
            #ffgrid_batch = random.choice(keys_next[0],train_data2['pos'],shape=(self.c.optimization_init_kwargs["f_batch"],))
            #ffval_batch = random.choice(keys_next[0],train_data2_,shape=(self.c.optimization_init_kwargs["f_batch"],))

            g_batch = jnp.stack([random.choice(keys_next[k+1], 
                                            grids['eqns'][arg], 
                                            shape=(self.c.optimization_init_kwargs["e_batch"],)) 
                                for k, arg in enumerate(list(all_params["domain"]["domain_range"].keys()))],axis=1)
            b_batches = []
            for b_key in all_params["domain"]["bound_keys"]:
                b_batch = jnp.stack([random.choice(keys_next[k+5], 
                                                grids[b_key][arg], 
                                                shape=(self.c.optimization_init_kwargs["e_batch"],)) 
                                    for k, arg in enumerate(list(all_params["domain"]["domain_range"].keys()))],axis=1)
                b_batches.append(b_batch)
            loss_factor = jnp.exp(-k*0.0001)
            lossval, model_states, dynamic_params = update(model_states, dynamic_params, static_params, loss_factor, g_batch, ffgrid_batch, ffval_batch, p_batch, v_batch, b_batches)
        
        
            self.report2(i, report_fn2, dynamic_params, all_params, loss_factor, p_batch, v_batch, g_batch, ffgrid_batch, ffval_batch, b_batch, valid_data, keys_iter[-1], self.c.optimization_init_kwargs["save_step"], model_fn)
            self.save_model(i, dynamic_params, all_params, self.c.optimization_init_kwargs["save_step"], model_fn)
        
    def save_model(self, i, dynamic_params, all_params, save_step, model_fns):
        model_save = (i % save_step == 0)
        if model_save:
            all_params["network"]["layers"] = dynamic_params
            model = Model(all_params["network"]["layers"], model_fns)
            serialised_model = to_state_dict(model)
            with open(self.c.model_out_dir + "saved_dic_"+str(i)+".pkl","wb") as f:
                pickle.dump(serialised_model,f)
        return

    def report1(self, i, report_fn, dynamic_params, all_params, p_batch, v_batch, g_batch, ffgrid_batch, ffval_batch, b_batch, valid_data, e_batch_key, save_step, model_fns):
        save_report = (i % save_step == 0)
        if save_report:
            all_params["network"]["layers"] = dynamic_params
            e_key = next(e_batch_key)
            e_batch_pos = random.choice(e_key, valid_data['pos'], shape = (self.c.optimization_init_kwargs["e_batch"],))
            e_batch_vel = random.choice(e_key, valid_data['vel'], shape = (self.c.optimization_init_kwargs["e_batch"],))
            if 'T' in valid_data.keys():
                e_batch_T = random.choice(e_key, valid_data['T'], shape = (self.c.optimization_init_kwargs["e_batch"],))
            v_pred = model_fns(all_params, e_batch_pos)
            print(all_params["data"]['u_ref'])
            u_error = jnp.sqrt(jnp.mean((all_params["data"]["u_ref"]*v_pred[:,0:1] - e_batch_vel[:,0:1])**2)/jnp.mean(e_batch_vel[:,0:1]**2))
            v_error = jnp.sqrt(jnp.mean((all_params["data"]["v_ref"]*v_pred[:,1:2] - e_batch_vel[:,1:2])**2)/jnp.mean(e_batch_vel[:,1:2]**2))
            w_error = jnp.sqrt(jnp.mean((all_params["data"]["w_ref"]*v_pred[:,2:3] - e_batch_vel[:,2:3])**2)/jnp.mean(e_batch_vel[:,2:3]**2))
            if v_pred.shape[1] == 5:
                T_error = jnp.sqrt(jnp.mean((all_params["data"]["T_ref"]*v_pred[:,4] - e_batch_T)**2)/jnp.mean(e_batch_T**2))
            
            Losses = report_fn(dynamic_params, all_params, ffgrid_batch, ffval_batch, p_batch, v_batch, b_batch, model_fns)
            if v_pred.shape[1] == 5:
                print(f"step_num : {i:<{12}} u_loss : {Losses[1]:<{12}.{5}} v_loss : {Losses[2]:<{12}.{5}} w_loss : {Losses[3]:<{12}.{5}} u_error : {u_error:<{12}.{5}} v_error : {v_error:<{12}.{5}} w_error : {w_error:<{12}.{5}} T_error : {T_error:<{12}.{5}}")
                with open(self.c.report_out_dir + "reports.txt", "a") as f:
                    f.write(f"{i:<{12}} {Losses[0]:<{12}.{5}} {Losses[1]:<{12}.{5}} {Losses[2]:<{12}.{5}} {Losses[3]:<{12}.{5}} {Losses[4]:<{12}.{5}} {Losses[5]:<{12}.{5}} {Losses[6]:<{12}.{5}} {Losses[7]:<{12}.{5}} {Losses[8]:<{12}.{5}} {u_error:<{12}.{5}} {v_error:<{12}.{5}} {w_error:<{12}.{5}} {T_error:<{12}.{5}}\n")
            else:
                print(f"step_num : {i:<{12}} u_loss : {Losses[1]:<{12}.{5}} v_loss : {Losses[2]:<{12}.{5}} w_loss : {Losses[3]:<{12}.{5}} u_error : {u_error:<{12}.{5}} v_error : {v_error:<{12}.{5}} w_error : {w_error:<{12}.{5}}")
                with open(self.c.report_out_dir + "reports.txt", "a") as f:
                    f.write(f"{i:<{12}} {Losses[0]:<{12}.{5}} {Losses[1]:<{12}.{5}} {Losses[2]:<{12}.{5}} {Losses[3]:<{12}.{5}} {Losses[4]:<{12}.{5}} {Losses[5]:<{12}.{5}} {Losses[6]:<{12}.{5}} {Losses[7]:<{12}.{5}} {0.0:<{12}.{5}} {0.0:<{12}.{5}} {0.0:<{12}.{5}} {0.0:<{12}.{5}} {0.0:<{12}.{5}} {u_error:<{12}.{5}} {v_error:<{12}.{5}} {w_error:<{12}.{5}} {0.0:<{12}.{5}}\n")
            f.close()
        return

    
    def report2(self, i, report_fn, dynamic_params, all_params, loss_factor, p_batch, v_batch, g_batch, ffgrid_batch, ffval_batch, b_batch, valid_data, e_batch_key, save_step, model_fns):
        save_report = (i % save_step == 0)
        if save_report:
            all_params["network"]["layers"] = dynamic_params
            e_key = next(e_batch_key)
            e_batch_pos = random.choice(e_key, valid_data['pos'], shape = (self.c.optimization_init_kwargs["e_batch"],))
            e_batch_vel = random.choice(e_key, valid_data['vel'], shape = (self.c.optimization_init_kwargs["e_batch"],))
            if 'T' in valid_data.keys():
                e_batch_T = random.choice(e_key, valid_data['T'], shape = (self.c.optimization_init_kwargs["e_batch"],))
            v_pred = model_fns(all_params, e_batch_pos)
            print(all_params["data"]['u_ref'])
            u_error = jnp.sqrt(jnp.mean((all_params["data"]["u_ref"]*v_pred[:,0:1] - e_batch_vel[:,0:1])**2)/jnp.mean(e_batch_vel[:,0:1]**2))
            v_error = jnp.sqrt(jnp.mean((all_params["data"]["v_ref"]*v_pred[:,1:2] - e_batch_vel[:,1:2])**2)/jnp.mean(e_batch_vel[:,1:2]**2))
            w_error = jnp.sqrt(jnp.mean((all_params["data"]["w_ref"]*v_pred[:,2:3] - e_batch_vel[:,2:3])**2)/jnp.mean(e_batch_vel[:,2:3]**2))
            if v_pred.shape[1] == 5:
                T_error = jnp.sqrt(jnp.mean((all_params["data"]["T_ref"]*v_pred[:,4] - e_batch_T)**2)/jnp.mean(e_batch_T**2))

            Losses = report_fn(dynamic_params, all_params, loss_factor, g_batch, ffgrid_batch, ffval_batch, p_batch, v_batch, b_batch, model_fns)
            if v_pred.shape[1] == 5:
                print(f"step_num : {i:<{12}} u_loss : {Losses[1]:<{12}.{5}} v_loss : {Losses[2]:<{12}.{5}} w_loss : {Losses[3]:<{12}.{5}} u_error : {u_error:<{12}.{5}} v_error : {v_error:<{12}.{5}} w_error : {w_error:<{12}.{5}} T_error : {T_error:<{12}.{5}}")
                with open(self.c.report_out_dir + "reports.txt", "a") as f:
                    f.write(f"{i:<{12}} {Losses[0]:<{12}.{5}} {Losses[1]:<{12}.{5}} {Losses[2]:<{12}.{5}} {Losses[3]:<{12}.{5}} {Losses[4]:<{12}.{5}} {Losses[5]:<{12}.{5}} {Losses[6]:<{12}.{5}} {Losses[7]:<{12}.{5}} {Losses[8]:<{12}.{5}} {Losses[9]:<{12}.{5}} {Losses[10]:<{12}.{5}} {Losses[11]:<{12}.{5}} {Losses[12]:<{12}.{5}} {u_error:<{12}.{5}} {v_error:<{12}.{5}} {w_error:<{12}.{5}} {T_error:<{12}.{5}}\n")
            else:
                print(f"step_num : {i:<{12}} u_loss : {Losses[1]:<{12}.{5}} v_loss : {Losses[2]:<{12}.{5}} w_loss : {Losses[3]:<{12}.{5}} u_error : {u_error:<{12}.{5}} v_error : {v_error:<{12}.{5}} w_error : {w_error:<{12}.{5}}")
                with open(self.c.report_out_dir + "reports.txt", "a") as f:
                    f.write(f"{i:<{12}} {Losses[0]:<{12}.{5}} {Losses[1]:<{12}.{5}} {Losses[2]:<{12}.{5}} {Losses[3]:<{12}.{5}} {Losses[4]:<{12}.{5}} {Losses[5]:<{12}.{5}} {Losses[6]:<{12}.{5}} {Losses[7]:<{12}.{5}} {Losses[8]:<{12}.{5}} {Losses[9]:<{12}.{5}} {Losses[10]:<{12}.{5}} {Losses[11]:<{12}.{5}} {0.0:<{12}.{5}} {u_error:<{12}.{5}} {v_error:<{12}.{5}} {w_error:<{12}.{5}} {0.0:<{12}.{5}}\n")
            f.close()
        return

#%%
if __name__=="__main__":
    from domain import *
    from trackdata import *
    from network import *
    from constants import *
    from problem import *
    from equation import *
    from txt_reader import *
    import argparse
    
    parser = argparse.ArgumentParser(description='TBL_PINN')
    parser.add_argument('-n', '--name', type=str, help='run name', default='HIT_k1')
    parser.add_argument('-c', '--config', type=str, help='configuration', default='test_txt')
    args = parser.parse_args()
    cur_dir = os.getcwd()
    input_txt = cur_dir + '/' + args.config + '.txt' 
    data = parse_tree_structured_txt(input_txt)
    c = Constants(**data)

    run = PINN(c)
    run.train()