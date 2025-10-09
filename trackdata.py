#%%
import numpy as np
from glob import glob
import os
class Database:
    @staticmethod
    def init_parmas(path, s_range, t_range, track_limit):
        raise NotImplementedError
    @staticmethod
    def data_load(filename):
        raise NotImplementedError
    @staticmethod
    def track_filter(track):
        raise NotImplementedError
    @staticmethod
    def domain_filter(s_trange):
        raise NotImplementedError
    @staticmethod
    def data_split(data, ratio):
        raise NotImplementedError
    
class Data(Database):
    @staticmethod
    #def init_params(path, domain_range, timeskip, track_limit, frequency, data_keys, viscosity):
    def init_params(**kwargs):
        data_params = {}
        for key, value in kwargs.items():
            data_params[key] = value
        return data_params

    @staticmethod
    def data_load_npy(filename, data_keys):
        data = np.load(filename)
        cols = {'pos':4, 'vel':3, 'p':1, 'T':1, 'acc':3}
        required_keys = ['pos', 'vel']
        for required_key in required_keys:
            if required_key not in data_keys:
                raise ValueError(f"Required '{required_key}' key is not in data_keys.")
        all_data = {}
        idx = 0
        for col in cols.keys():
            if col in data_keys:
                all_data[col] = data[:,idx:idx+cols[col]]
                idx += cols[col]
        return all_data

    @staticmethod
    def domain_filter(all_data_, data_keys, domain_range):
        index = np.where((all_data_['pos'][:,1]>=domain_range['x'][0])&(all_data_['pos'][:,1]<=domain_range['x'][1])&
                         (all_data_['pos'][:,2]>=domain_range['y'][0])&(all_data_['pos'][:,2]<=domain_range['y'][1])&
                         (all_data_['pos'][:,3]>=domain_range['z'][0])&(all_data_['pos'][:,3]<=domain_range['z'][1]))
        all_data_ = {data_keys[i]:all_data_[data_keys[i]][index[0],:] for i in range(len(data_keys))}
        return all_data_
    
    @staticmethod
    def input_normalize(all_params, data):
        domain_range = all_params["domain"]["domain_range"]
        arg_keys = ['t', 'x', 'y', 'z']
        for i in range(data['pos'].shape[1]):
            data['pos'][:,i] = data['pos'][:,i]/domain_range[arg_keys[i]][1]
        return data

    @staticmethod
    def output_normalize(all_params, data):
        vel_ref_keys = ['u_ref', 'v_ref', 'w_ref']
        vel_ref = {vel_ref_keys[i]:np.max(np.abs(data['vel'][:,i:i+1])) for i in range(len(vel_ref_keys))}
        vel_ref['p_ref'] = vel_ref['u_ref']
        vel_ref['T_ref'] = 0.5
        all_params["data"].update(vel_ref)
        return all_params
    """
    @staticmethod
    def bound_sampler(all_params, domain_range, seed, num, time):
        bound_keys = all_params["data"]["bound_keys"]
        data_keys = all_params["data"]["data_keys"]
        np.random.seed(seed)
        seeds = np.random.randint(0,1000,2*len(bound_keys))
        total_bound = {}
        for i,key in enumerate(bound_keys.keys()):
            np.random.seed(seeds[i])
            nx = np.random.uniform(*domain_range['x'], bound_keys[key][-1])
            np.random.seed(seeds[2*i])
            ny = np.random.uniform(*domain_range['y'], bound_keys[key][-1])
            np.random.seed(seeds[3*i])
            nz = np.random.uniform(*domain_range['z'], bound_keys[key][-1])
 
            if key == 'x':
                temp = []
                if bound_keys[key][0]:
                    temp_ = np.column_stack([np.zeros(bound_keys[key][-1])+time,
                                            np.zeros(bound_keys[key][-1]+domain_range['x'][0]),
                                            ny,
                                            nz])
                    temp.append(temp_)
                if bound_keys[key][1]:
                    temp_ = np.column_stack([np.zeros(bound_keys[key][-1])+time,
                                            np.zeros(bound_keys[key][-1]+domain_range['x'][1]),
                                            ny,
                                            nz])
                    temp.append(temp_)
                if len(temp) > 1:
                    total_bound[bound_keys[key]] = np.concatenate(temp, 0)
                elif len(temp) > 0:
                    total_bound[bound_keys[key]] = temp[0]
                else:
                    total_bound[bound_keys[key]] = None
            if key == 'y':
                temp = []
                if bound_keys[key][0]:
                    temp_ = np.column_stack([np.zeros(bound_keys[key][-1])+time,
                                            nx,
                                            np.zeros(bound_keys[key][-1]+domain_range['y'][0]),
                                            nz])
                    temp.append(temp_)
                if bound_keys[key][1]:
                    temp_ = np.column_stack([np.zeros(bound_keys[key][-1])+time,
                                            nx,
                                            np.zeros(bound_keys[key][-1]+domain_range['y'][1]),
                                            nz])
                    temp.append(temp_)
                if len(temp) > 1:
                    total_bound[bound_keys[key]] = np.concatenate(temp, 0)
                elif len(temp) > 0:
                    total_bound[bound_keys[key]] = temp[0]
                else:
                    total_bound[bound_keys[key]] = None
            if key == 'z':
                temp = []
                if bound_keys[key][0]:
                    temp_ = np.column_stack([np.zeros(bound_keys[key][-1])+time,
                                            nx,
                                            ny,
                                            np.zeros(bound_keys[key][-1]+domain_range['z'][0])])
                    temp.append(temp_)
                if bound_keys[key][1]:
                    temp_ = np.column_stack([np.zeros(bound_keys[key][-1])+time,
                                            nx,
                                            ny,
                                            np.zeros(bound_keys[key][-1]+domain_range['z'][1])])
                    temp.append(temp_)
                if len(temp) > 1:
                    total_bound[bound_keys[key]] = np.concatenate(temp, 0)
                elif len(temp) > 0:
                    total_bound[bound_keys[key]] = temp[0]
                else:
                    total_bound[bound_keys[key]] = None
        if 'acc' in data_keys:
            for key in bound_keys.keys():
                if total_bound[key]:
                    total_bound[key] = np.concatenate([total_bound[key],
                                                        np.zeros((total_bound[key].shape[0],3))],1)
        if 'T' in data_keys:
            for key in bound_keys.keys():
                if total_bound[key]:
                    total_bound[key] = np.concatenate([total_bound[key],
                                                        np.zeros((total_bound[key].shape[0],1))],1)
        
            if bound_keys[key][0]:
                total_bound[bound_key] = 
        for i, bound_key in enumerate(bound_keys):
            np.random.seed(seeds[i])
            nx = np.random.uniform(*domain_range['x'], num[i])
            np.random.seed(seeds[2*i])
            ny = np.random.uniform(*domain_range['y'], num[i])
            
            if 'acc' in all_params["data"]["data_keys"]:
                if 'T' in all_params["data"]["data_keys"]:
                    if bound_key == 'bczl':
                        total_bound[bound_key] = np.column_stack([np.zeros(num[i])+time,
                                                                np.random.uniform(*domain_range['x'], num[i]),
                                                                np.random.uniform(*domain_range['y'], num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i])-0.5])
                    elif bound_key == 'bczu':
                        total_bound[bound_key] = np.column_stack([np.zeros(num[i])+time,
                                                                np.random.uniform(*domain_range['x'], num[i]),
                                                                np.random.uniform(*domain_range['y'], num[i]),
                                                                np.ones(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i])+0.5])
                else:
                    if bound_key == 'bczl':
                        total_bound[bound_key] = np.column_stack([np.zeros(num[i])+time,
                                                                np.random.uniform(*domain_range['x'], num[i]),
                                                                np.random.uniform(*domain_range['y'], num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i])])
                    elif bound_key == 'bczu':
                        total_bound[bound_key] = np.column_stack([np.zeros(num[i])+time,
                                                                np.random.uniform(*domain_range['x'], num[i]),
                                                                np.random.uniform(*domain_range['y'], num[i]),
                                                                np.ones(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i])])
            else:
                if 'T' in all_params["data"]["data_keys"]:
                    if bound_key == 'bczl':
                        total_bound[bound_key] = np.column_stack([np.zeros(num[i])+time,
                                                                np.random.uniform(*domain_range['x'], num[i]),
                                                                np.random.uniform(*domain_range['y'], num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i])-0.5])
                    elif bound_key == 'bczu':
                        total_bound[bound_key] = np.column_stack([np.zeros(num[i])+time,
                                                                np.random.uniform(*domain_range['x'], num[i]),
                                                                np.random.uniform(*domain_range['y'], num[i]),
                                                                np.ones(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i]),
                                                                np.zeros(num[i])+0.5])
        
        return total_bound
    """

    @staticmethod
    def train_data(all_params):
        cur_dir = os.getcwd()
        path = all_params["data"]["path"]
        domain_range = all_params["domain"]["domain_range"]
        data_keys = all_params["data"]["data_keys"]
        #bound_keys = all_params["data"]["bound_keys"]

        filenames = sorted(glob(os.path.dirname(cur_dir)+path+'*.npy'))

        datas = {data_keys[i]:[] for i in range(len(data_keys))}

        seed_number = np.arange(0,1000)
        np.random.seed(42)
        seeds = np.random.choice(seed_number, len(filenames))

        for t, filename in enumerate(filenames):
            all_data_ = Data.data_load_npy(filename, data_keys)
            """    
            if "bound_keys" in list(all_params["data"].keys()):
                np.random.seed(seeds[t])
                total_bound = Data.bound_sampler(all_params, domain_range, seeds[t], all_params["data"]['bound_numbers'], time = all_data_['pos'][0,0])
                for bound_key in bound_keys:
                    all_data_['pos'] = np.concatenate([all_data_['pos'], total_bound[bound_key][:,0:4]], 0)
                    all_data_['vel'] = np.concatenate([all_data_['vel'], total_bound[bound_key][:,4:7]], 0)
                    if 'acc' in data_keys:
                        all_data_['acc'] = np.concatenate([all_data_['acc'], total_bound[bound_key][:,7:10]], 0)
                        if 'T' in data_keys:
                            all_data_['T'] = np.concatenate([all_data_['T'], total_bound[bound_key][:,10:11]], 0)
                    else:
                        if 'T' in data_keys:
                            all_data_['T'] = np.concatenate([all_data_['T'], total_bound[bound_key][:,7:8]], 0)                        
            """
            for i in range(len(data_keys)): datas[data_keys[i]].append(all_data_[data_keys[i]])
        for j in range(len(data_keys)): datas[data_keys[j]] = np.concatenate(datas[data_keys[j]], 0, dtype=np.float64)
        datas = Data.domain_filter(datas, data_keys, domain_range)

        train_data = Data.input_normalize(all_params, datas)
        all_params = Data.output_normalize(all_params, train_data)
        return train_data, all_params

    @staticmethod
    def ff_data(all_params):
        cur_dir = os.getcwd()
        path = all_params["data"]["path_ff"]
        domain_range = all_params["domain"]["domain_range"]
        data_keys = all_params["data"]["data_keys"]
        #bound_keys = all_params["data"]["bound_keys"]

        filenames = sorted(glob(os.path.dirname(cur_dir)+path+'*.npy'))

        datas = {data_keys[i]:[] for i in range(len(data_keys))}

        seed_number = np.arange(0,10000)
        np.random.seed(42)
        seeds = np.random.choice(seed_number, len(filenames))

        for t, filename in enumerate(filenames):
            all_data_ = Data.data_load_npy(filename, data_keys)

            for i in range(len(data_keys)): datas[data_keys[i]].append(all_data_[data_keys[i]])
        for j in range(len(data_keys)): datas[data_keys[j]] = np.concatenate(datas[data_keys[j]], 0, dtype=np.float64)
        datas = Data.domain_filter(datas, data_keys, domain_range)
        train_data = Data.input_normalize(all_params, datas)
        all_params = Data.output_normalize(all_params, train_data)
        return train_data


if __name__ == "__main__":
    from domain import *
    all_params = {"data":{}, "domain":{}}

    cur_dir = os.getcwd()
    #path = '/RBC_G8_DNS/npdata/lv6_xbound/'
    path = '/ETFS/HIT/train_data/lv1/'
    data_keys = ['pos', 'vel', 'acc', 'p']
    viscosity = 15*10**(-6)

    domain_range = {'t':(0,0.04), 'x':(0,0.1), 'y':(0,0.1), 'z':(0,0.1)}
    grid_size = [51, 200, 200, 200]
    bound_keys = ['ic', 'bcxu', 'bcxl', 'bcyu', 'bcyl', 'bczu', 'bczl']
    u_ref = 1.5
    v_ref = 1.5
    w_ref = 0.9
    p_ref = 1.5
    all_params["data"] = Data.init_params(path = path, 
                                          data_keys = data_keys, 
                                          viscosity = viscosity,
                                          u_ref = u_ref,
                                          v_ref = v_ref,
                                          w_ref = w_ref,
                                          p_ref = p_ref)
    all_params["domain"] = Domain.init_params(domain_range = domain_range, 
                                              bound_keys = bound_keys,
                                              grid_size = grid_size)
    
    train_data, all_params = Data.train_data(all_params)
