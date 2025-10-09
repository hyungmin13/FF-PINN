#%%
import numpy as np
import jax.numpy as jnp
import os
from glob import glob
class Domainbase:
    @staticmethod
    def init_params(*args):
        raise NotImplementedError

    @staticmethod
    def bound_sampler(all_params,grids):
        raise NotImplementedError
    
    @staticmethod
    def normalize(all_params, grids):
        raise NotImplementedError

    @staticmethod
    def sampler(all_params):
        raise NotImplementedError

    
class Domain(Domainbase):
    @staticmethod
    def init_params(**kwargs):
        domain_params = {}
        for key, value in kwargs.items():
            domain_params[key] = value
        return domain_params

    @staticmethod
    def bound_sampler(all_params, grids):
        bound_keys = all_params["domain"]["bound_keys"]
        arg_keys = ['t', 'x', 'y', 'z']
        total_bound = {bound_keys[i]:{arg_keys[j]:[] for j in range(len(arg_keys))} for i in range(len(bound_keys))}
        for i, bound_key in enumerate(bound_keys):
            for j in range(len(arg_keys)):
                total_bound[bound_key][arg_keys[j]] = grids['eqns'][arg_keys[j]]
            for arg_key in arg_keys:
                if arg_key in bound_key:
                    if 'u' in bound_key:
                        total_bound[bound_key][arg_key] = np.array([grids['eqns'][arg_key][-1]])
                    else:
                        total_bound[bound_key][arg_key] = np.array([grids['eqns'][arg_key][0]])
            if 'ic' in bound_key:
                total_bound[bound_key]['t'] = grids['eqns']['t'][0]
        grids.update(total_bound)
        return grids
    
    @staticmethod
    def normalize(all_params, grids):
        domain_range = all_params["domain"]["domain_range"]
        key_list = list(grids.keys())
        arg_keys = ['t', 'x', 'y', 'z']
        for i in range(len(key_list)):
            for arg_key in arg_keys:
                grids[key_list[i]][arg_key] = grids[key_list[i]][arg_key]/domain_range[arg_key][1]
            #grids[key_list[i]]['t'] = grids[key_list[i]]['t']*domain_range[arg_key][1]*frequency
        return grids

    @staticmethod
    def sampler(all_params):
        domain_range = all_params["domain"]["domain_range"]
        grid_size = all_params["domain"]["grid_size"]
        path = all_params["data"]['path']
        cur_dir = os.getcwd()
        filenames = sorted(glob(os.path.dirname(cur_dir)+path+'*.npy'))
        t = []
        for filename in filenames:
            temp = np.load(filename)
            t.append(temp[0,0])
        t = np.array(t)
        arg_keys = ['t', 'x', 'y', 'z']
        grids = {'eqns':{arg_keys[j]:[] for j in range(len(arg_keys))}}
        grids['eqns']['t'] = t
        for i in range(len(arg_keys)-1):
            grids['eqns'][arg_keys[i+1]] = np.linspace(domain_range[arg_keys[i+1]][0], domain_range[arg_keys[i+1]][1], grid_size[i])
        grids = Domain.bound_sampler(all_params, grids)
        grids = Domain.normalize(all_params, grids)
        all_params["domain"]["in_min"] = jnp.array([[domain_range['t'][0], domain_range['x'][0], domain_range['y'][0], domain_range['z'][0]]])
        all_params["domain"]["in_max"] = jnp.array([[domain_range['t'][1], domain_range['x'][1], domain_range['y'][1], domain_range['z'][1]]])
        return grids, all_params
    


if __name__ == "__main__":
    from trackdata import *
    all_params = {"domain":{}, "data":{}}
    path = '/RBC_G8_DNS/npdata/lv6_xbound/'
    data_keys = ['pos', 'vel', 'T']
    viscosity = 15*10**(-6)
    all_params["data"] = Data.init_params(path = path, data_keys = data_keys, viscosity = viscosity)
    
    domain_range = {'t':(0,7.4), 'x':(0,8), 'y':(0,8), 'z':(0,1)}
    grid_size = [51, 200, 200, 200]
    bound_keys = ['ic', 'bcxu', 'bcxl', 'bcyu', 'bcyl', 'bczu', 'bczl']
    all_params["domain"] = Domain.init_params(domain_range = domain_range, grid_size = grid_size, bound_keys = bound_keys)
    
    grids, all_params = Domain.sampler(all_params)

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    mesh_xyz = np.meshgrid(grids['eqns']['x'], grids['eqns']['y'], grids['eqns']['z'], indexing='ij')
    
    print(grids['eqns']['t'])

    fig, axes = plt.subplots(figsize=(8,8), nrows=2, ncols=2)
    for i in range(3):
        j, k = (i//2, i%2)
        im = axes[j,k].imshow(mesh_xyz[i][0,:,:],cmap='jet')
        divider = make_axes_locatable(axes[j,k])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    plt.show()
