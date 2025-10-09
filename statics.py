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
import numpy as np
import os
from glob import glob
from numpy.fft import fft
#%%
def shear_stress(streamwise_vel, normal_axis):
    return (streamwise_vel[:,1,:,:]-streamwise_vel[:,0,:,:])/(normal_axis[0,1,0,0]-normal_axis[0,0,0,0])
def friction_vel(streamwise_vel, normal_axis, viscosity):
    return np.sqrt(viscosity*np.mean(streamwise_vel[:,1,:,:]-streamwise_vel[:,0,:,:])/(normal_axis[1,0,0]-normal_axis[0,0,0]))
def viscous_length(streamwise_vel, normal_axis, viscosity):
    return viscosity/np.sqrt(viscosity*np.mean(streamwise_vel[:,1,:,:]-streamwise_vel[:,0,:,:])/(normal_axis[1,0,0]-normal_axis[0,0,0]))
def y_plus(streamwise_vel, normal_axis, viscosity):
    return normal_axis[:,0,0]/(viscosity/np.sqrt(viscosity*np.mean(streamwise_vel[:,1,:,:]-streamwise_vel[:,0,:,:])/(normal_axis[1,0,0]-normal_axis[0,0,0])))
def u_plus(streamwise_vel, streamwise_vel_ground, normal_axis, viscosity):
    return np.mean(np.mean(np.mean(streamwise_vel,0),1),1)/(np.sqrt(viscosity*np.mean(streamwise_vel_ground[:,1,:,:]-streamwise_vel_ground[:,0,:,:])/(normal_axis[1,0,0]-normal_axis[0,0,0])))
#%%
u_tau = 15*10**(-6)/36.2/10**(-6)
delta = 36.2*10**(-6)
u_ref = 4.9968*10**(-2)/u_tau
x_ref = 1.0006*10**(-3)/delta
friction_vel_true = 4.9968*10**(-2)
viscous_length_true = 1.0006*10**(-3)
#%%
key = ['u', 'v', 'w', 'p']
cur_dir = os.getcwd()
path  = os.path.dirname(cur_dir) + '/ETFS/TBL_syn/npyresult/TBL_SOAP_k1'
filenames = sorted(glob(path + '/*.npy'))
pred_data = np.concatenate([np.load(filenames[i]).reshape(1,213,61,141,-1) for i in tqdm(range(len(filenames)-48))],0)
pred_data[:,:,:,:,1:4] = pred_data[:,:,:,:,1:4]*x_ref
pred_data[:,:,:,:,4:] = pred_data[:,:,:,:,4:]*u_ref
pred_data[:,:,:,:,7] = pred_data[:,:,:,:,7] - np.mean(pred_data[:,:,:,:,7])
path_ground = os.path.dirname(cur_dir) + '/ETFS/TBL_syn/ground_turb'
filenames_ground = glob(path_ground + '/*.npy')
ground_data = np.concatenate([np.swapaxes(np.load(filenames_ground[i]).reshape(1,213, 141, 61, 8),2,3) for i in range(len(filenames_ground)-48)],0)
ground_data[:,:,:,:,1:4] = ground_data[:,:,:,:,1:4]*x_ref
ground_data[:,:,:,:,4:] = ground_data[:,:,:,:,4:]*u_ref
ground_data[:,:,:,:,7] = ground_data[:,:,:,:,7] - np.mean(ground_data[:,:,:,:,7])
#%%
plt.imshow(pred_data[0, 10,:,:,4],cmap='jet')
plt.colorbar()
plt.show()
plt.imshow(ground_data[0, 10,:,:,4],cmap='jet')
plt.colorbar()
plt.show()
#%%

#%%
shear_stress_pred = np.mean(pred_data[:,0,:,:,16])/x_ref
friction_vel_pred = np.sqrt(shear_stress_pred*5*10**(-5))
viscous_length_pred = 5*10**(-5)/friction_vel_pred
y_plus_pred = pred_data[0,:,0,0,3]/viscous_length_pred
u_plus_pred = np.mean(np.mean(np.mean(pred_data[i][:,:,:,4].reshape(1,213,141,61),0),1),1)/friction_vel_pred
#%%
shear_stress_ground = np.mean(shear_stress(ground_data[:,:,:,:,4],ground_data[:,:,:,:,3]))
friction_vel_ground = np.sqrt(5*10**(-5)*shear_stress_ground)
viscous_length_ground = 5*10**(-5)/friction_vel_ground
y_plus_ground = ground_data[0][:,0,0,3]/viscous_length_ground
u_plus_ground = np.mean(np.mean(np.mean(np.concatenate([ground_data[i][:,:,:,4].reshape(1,213,141,61) for i in range(len(ground_data))],0),0),1),1)/friction_vel_ground


#%%
print('shear stress : ', shear_stress_ground, shear_stress_pred)
print('friction velocity : ', friction_vel_ground, friction_vel_pred)
print('viscous length : ', viscous_length_ground, viscous_length_pred)
print('friction Re : ', friction_vel_ground/5/10**(-5), friction_vel_pred/5/10**(-5))
#%% y+ - U+ plot
plt.plot(y_plus_ground, u_plus_ground,'k-',label='Ground')
plt.plot(y_plus_pred, u_plus_pred,'r--',label='PINNs')
plt.plot(y_plus_ground[:30], y_plus_ground[:30],'b.',label='$U^+$ = $y^+$')
plt.legend()
plt.xlabel('$Y^+$')
plt.ylabel('$U^+$')
plt.title('mean velociy profile in viscous units')
plt.xscale('log')
plt.show()

#%% velocity covariance curve
key_fluc = ['u_fluc', 'v_fluc', 'w_fluc']
#fluc_ground = {key_fluc[i]:ground_data[key[i]]-np.mean(np.mean(np.mean(ground_data[key[i]],0),1),1).reshape(1,-1,1,1) for i in range(3)}
fluc_ground = {key_fluc[j]:ground_data[:,:,:,:,j+4]-np.mean(np.mean(np.mean(ground_data[:,:,:,:,j+4],0),1),1).reshape(1,-1,1,1) for j in range(3)}
fluc_pred = {key_fluc[j]:pred_data[:,:,:,:,j+4]-np.mean(np.mean(np.mean(pred_data[:,:,:,:,j+4],0),1),1).reshape(1,-1,1,1) for j in range(3)}

label_key = ['uu_ground','uu_pred','vv_ground','vv_pred','ww_ground','ww_pred']
color_key = ['k', 'b', 'r']
for i in range(len(key_fluc)):
    plt.plot(y_plus_ground,np.mean(np.mean(np.mean(fluc_ground[key_fluc[i]]*fluc_ground[key_fluc[i]],0),1),1)/friction_vel_true**2, color_key[i]+'-',label=label_key[2*i])
    plt.plot(y_plus_pred,np.mean(np.mean(np.mean(fluc_pred[key_fluc[i]]*fluc_pred[key_fluc[i]],0),1),1)/friction_vel_true**2, color_key[i]+'--',label=label_key[2*i+1])
plt.xlabel('$Y^+$')
plt.ylabel('$u_i$$u_j$')
plt.xscale('log')
plt.title('velocity covariance in viscous units')
#plt.yscale('log')
plt.legend()
plt.show()

#%% 
p_ground_mean = np.mean(np.mean(np.mean(ground_data[:,:,:,:,7],0),1),1)
plt.plot(y_plus_ground,(np.mean(np.mean(np.mean(ground_data[:,:,:,:,7],0),1),1)-np.mean(np.mean(np.mean(ground_data[:,:,:,:,7],0),1),1)[0])/friction_vel_ground*2, 'k-',label='Ground')
plt.plot(y_plus_ground,(np.mean(np.mean(np.mean(pred_data[:,:,:,:,7],0),1),1)-np.mean(np.mean(np.mean(pred_data[:,:,:,:,7],0),1),1)[0])/friction_vel_pred*2, 'r--',label='PINNs')
plt.xlabel('$Y^+$')
plt.ylabel('$P^+$')
plt.legend()
plt.title('Mean pressure profile in viscous units')
plt.show()
#%%
n_ = [20, 55, 190]
y_ = y_plus_ground[n_]
print(y_)
Rij_key = ['Ruu', 'Rww', 'Rvv']

Rij_g = {}
Rij_p = {}
for n in n_:
    Rij_g[str(y_plus_ground[n])] = {}
    for j, Rij_key_ in enumerate(Rij_key):
        Rij_g[str(y_plus_ground[n])][Rij_key_] = []
        for i in range(ground_data[:,:,:,:,4].shape[2]):
            Rij_g_ = np.mean(fluc_ground[key_fluc[j]][:,n,:,:]*np.roll(fluc_ground[key_fluc[j]][:,n,:,:],i,axis=1))
            Rij_g[str(y_plus_ground[n])][Rij_key_].append(Rij_g_)
for n in n_:
    Rij_p[str(y_plus_ground[n])] = {}
    for j, Rij_key_ in enumerate(Rij_key):
        Rij_p[str(y_plus_ground[n])][Rij_key_] = []
        for i in range(pred_data[:,:,:,:,4].shape[2]):
            Rij_p_ = np.mean(fluc_pred[key_fluc[j]][:,n,:,:]*np.roll(fluc_pred[key_fluc[j]][:,n,:,:],i,axis=1))
            Rij_p[str(y_plus_ground[n])][Rij_key_].append(Rij_p_)



Eij_key = ['Euu', 'Eww', 'Evv']
Eij_g = {}
Eij_p = {}

for n in n_:
    Eij_g[str(y_plus_ground[n])] = {}
    Eij_p[str(y_plus_ground[n])] = {}
    for j, Eij_key_ in enumerate(Eij_key):
        Eij_g[str(y_plus_ground[n])][Eij_key_] = fft(Rij_g[str(y_plus_ground[n])][Rij_key[j]])
        Eij_p[str(y_plus_ground[n])][Eij_key_] = fft(Rij_p[str(y_plus_ground[n])][Rij_key[j]])


frequency_ground = np.fft.fftfreq(len(Eij_g[str(y_plus_ground[n_[0]])]['Euu']), ground_data[:,:,:,:,3][0,1,0,0]-ground_data[:,:,:,:,3][0,0,0,0])
frequency_pred = np.fft.fftfreq(len(Eij_p[str(y_plus_ground[n_[0]])]['Euu']), pred_data[:,:,:,:,3][0,1,0,0]-pred_data[:,:,:,:,3][0,0,0,0])

color_key = ['r', 'b', 'g']
shape_key = ['-', '--']
fig, axes = plt.subplots(1,3,figsize=(12,6))
for i in range(len(list(Eij_g.keys()))):
    m = i//2
    n = i%2
    for j in range(len(list(Eij_g[str(y_plus_ground[n_[0]])].keys()))):
        a = axes[i].plot(frequency_ground[:30],4*Eij_g[str(y_plus_ground[n_[i]])][Eij_key[j]][:30], color_key[j]+shape_key[0], label= Eij_key[j]+'_ground')
        b = axes[i].plot(frequency_pred[:30],4*Eij_p[str(y_plus_ground[n_[i]])][Eij_key[j]][:30], color_key[j]+shape_key[1], label= Eij_key[j]+'_PINNs')
    #c = axes[m,n].plot(frequency[:69], frequency[:70]**(-5/3), label='-5/3')
    #d = axes[m,n].plot(frequency[:70], frequency[:70]**(-1), label='-1')
    axes[i].set_xscale('log')
    axes[i].set_yscale('log')
    axes[0].set_xlabel('$k_x$')
    axes[0].set_ylabel('$E_i$$_j$')
    axes[i].set_title('$Y^+$ = ' + str(y_plus_ground[n_[i]]))
    axes[i].legend()
plt.show()

#%%
n_ = [20, 55, 190]
y_ = y_plus_ground[n_]
Rij_key = ['Ruu', 'Rww', 'Rvv']

Rij_g = {}
Rij_p = {}
for n in n_:
    Rij_g[str(y_plus_ground[n])] = {}
    for j, Rij_key_ in enumerate(Rij_key):
        Rij_g[str(y_plus_ground[n])][Rij_key_] = []
        for i in range(ground_data[:,:,:,:,4].shape[3]):
            Rij_g_ = np.mean(fluc_ground[key_fluc[j]][:,n,:,:]*np.roll(fluc_ground[key_fluc[j]][:,n,:,:],i,axis=2))
            Rij_g[str(y_plus_ground[n])][Rij_key_].append(Rij_g_)
for n in n_:
    Rij_p[str(y_plus_ground[n])] = {}
    for j, Rij_key_ in enumerate(Rij_key):
        Rij_p[str(y_plus_ground[n])][Rij_key_] = []
        for i in range(pred_data[:,:,:,:,4].shape[3]):
            Rij_p_ = np.mean(fluc_pred[key_fluc[j]][:,n,:,:]*np.roll(fluc_pred[key_fluc[j]][:,n,:,:],i,axis=2))
            Rij_p[str(y_plus_ground[n])][Rij_key_].append(Rij_p_)

Eij_key = ['Euu', 'Eww', 'Evv']
Eij_g = {}
Eij_p = {}

for n in n_:
    Eij_g[str(y_plus_ground[n])] = {}
    Eij_p[str(y_plus_ground[n])] = {}
    for j, Eij_key_ in enumerate(Eij_key):
        Eij_g[str(y_plus_ground[n])][Eij_key_] = fft(Rij_g[str(y_plus_ground[n])][Rij_key[j]])
        Eij_p[str(y_plus_ground[n])][Eij_key_] = fft(Rij_p[str(y_plus_ground[n])][Rij_key[j]])


frequency_ground = np.fft.fftfreq(len(Eij_g[str(y_plus_ground[n_[0]])]['Euu']), ground_data[:,:,:,:,3][0,1,0,0]-ground_data[:,:,:,:,3][0,0,0,0])
frequency_pred = np.fft.fftfreq(len(Eij_p[str(y_plus_ground[n_[0]])]['Euu']), pred_data[:,:,:,:,3][0,1,0,0]-pred_data[:,:,:,:,3][0,0,0,0])

color_key = ['r', 'b', 'g']
shape_key = ['-', '--']
fig, axes = plt.subplots(1,3,figsize=(12,6))
for i in range(len(list(Eij_g.keys()))):
    m = i//2
    n = i%2
    for j in range(len(list(Eij_g[str(y_plus_ground[n_[0]])].keys()))):
        a = axes[i].plot(frequency_ground[:30],4*Eij_g[str(y_plus_ground[n_[i]])][Eij_key[j]][:30], color_key[j]+shape_key[0], label= Eij_key[j]+'_ground')
        b = axes[i].plot(frequency_pred[:30],4*Eij_p[str(y_plus_ground[n_[i]])][Eij_key[j]][:30], color_key[j]+shape_key[1], label= Eij_key[j]+'_PINNs')
    #c = axes[m,n].plot(frequency[:70], frequency[:70]**(-5/3), label='-5/3')
    #d = axes[m,n].plot(frequency[:70], frequency[:70]**(-1), label='-1')
    axes[i].set_xscale('log')
    axes[i].set_yscale('log')
    axes[0].set_xlabel('$k_z$')
    axes[0].set_ylabel('$E_i$$_j$')
    axes[i].set_title('$Y^+$ = ' + str(y_plus_ground[n_[i]]))
    axes[i].legend()
plt.show()
