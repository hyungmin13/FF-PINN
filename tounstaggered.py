#%%
import os
from glob import glob
import pickle
import numpy as np
import h5netcdf as nc
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
# %%
cur_dir = os.getcwd()
geo = nc.File(os.path.dirname(cur_dir)+'/RBC_G8_DNS/ncdata/geo/'+'geometry.nc')
x_p = geo['x_p'][:]
x_ux = geo['x_ux'][:]
y_p = geo['y_p'][:]
y_uy = geo['y_uy'][:]
z_p = geo['z_p'][:]
z_uz = geo['z_uz'][:]
filenames = sorted(glob(os.path.dirname(cur_dir)+'/RBC_G8_DNS/ncdata/'+'*.nc'))
file0 = nc.File(filenames[0],'r')
t0 = file0['time'][()]
file1 = nc.File(filenames[1],'r')
t1 = file1['time'][()]
print(t0, t1)
#%%
print(file1['Ra'][()])
print(file1['Pr'][()])
print(np.sqrt(file1['Pr'][()]/file1['Ra'][()]))
print(np.sqrt(7/10**6))
print(np.sqrt(1/(file1['Pr'][()]*file1['Ra'][()])))
#%%
for j, filename in enumerate(filenames):
    file = nc.File(filename,'r')
    ux = file['ux'][:]
    uy = file['uy'][:]
    uz = file['uz'][:]
    T = file['temp'][:]
    t = file['time'][()] - t0
    zp, xp, yp = np.meshgrid(z_p, x_p, y_p, indexing='ij')
    points = np.array([zp.ravel(), xp.ravel(), yp.ravel()]).T
    zpu, xpu, ypu = np.meshgrid(z_p, x_ux[:-1], y_p, indexing='ij')
    cor = np.concatenate([zp.reshape(-1,1),xp.reshape(-1,1),yp.reshape(-1,1)],1)
    u_interp = RegularGridInterpolator((z_p, x_ux[:-1], y_p), ux[:,:-1,:], method='linear', bounds_error=False, fill_value=np.nan)
    v_interp = RegularGridInterpolator((z_p, x_p, y_uy[:-1]), uy[:,:,:-1], method='linear', bounds_error=False, fill_value=np.nan)
    w_interp = RegularGridInterpolator((z_uz[:-1], x_p, y_p), uz[:-1,:,:], method='linear', bounds_error=False, fill_value=np.nan)
    ulist = []
    vlist = []
    wlist = []
    for i in range(points.shape[0]//10000+1):
        u_new = u_interp(cor[i*10000:(i+1)*10000,:])
        v_new = v_interp(cor[i*10000:(i+1)*10000,:])
        z_new = w_interp(cor[i*10000:(i+1)*10000,:])
        ulist.append(u_new)
        vlist.append(v_new)
        wlist.append(z_new)
    u_new = np.concatenate(ulist, axis=0).reshape(98,542,542)
    v_new = np.concatenate(vlist, axis=0).reshape(98,542,542)
    w_new = np.concatenate(wlist, axis=0).reshape(98,542,542)
    newdata = np.concatenate([np.zeros(u_new.shape).reshape(-1,1)+t,
                            xp.reshape(-1,1),
                            yp.reshape(-1,1),
                            zp.reshape(-1,1),
                            u_new.reshape(-1,1),
                            v_new.reshape(-1,1),
                            w_new.reshape(-1,1),
                            T.reshape(-1,1)],axis=1)
    np.save(os.path.dirname(cur_dir)+'/RBC_G8_DNS/npdata/flow0' + str(170+j) + '.npy', newdata)
#%%
print(filenames[-1])
#%%
file = nc.File(os.path.dirname(cur_dir)+'/RBC_G8_DNS/ncdata/flow0183.nc','r')
ux = file['ux'][:]
uy = file['uy'][:]
uz = file['uz'][:]
T = file['temp'][:]
file2 = np.load(os.path.dirname(cur_dir)+'/RBC_G8_DNS/npdata/flow0183.npy')
#%%
filenames = sorted(glob(os.path.dirname(cur_dir)+'/RBC_G8_DNS/npdata/'+'*.npy'))
keys = np.arange(100)
index_ori = np.arange(file2.shape[0])
for i, filename in enumerate(filenames):
    np.random.seed(keys[i+28])
    index = np.random.choice(index_ori, size = 12*68*68)
    filenew = np.load(filename)[index,:]
    np.save(os.path.dirname(cur_dir)+'/RBC_G8_DNS/npdata/validate/flow0' + str(170+i) + '.npy', filenew)
#%%
filenames = sorted(glob(os.path.dirname(cur_dir)+'/RBC_G8_DNS/npdata/'+'*.npy'))
keys = np.arange(len(filenames))
index_ori = np.arange(file2.shape[0])
for i, filename in enumerate(filenames):
    np.random.seed(keys[i])
    index = np.random.choice(index_ori, size = 25*68*68)
    filenew = np.load(filename)[index,:]
    np.save(os.path.dirname(cur_dir)+'/RBC_G8_DNS/npdata/lv6/flow0' + str(170+i) + '.npy', filenew)
#%%
filenames = sorted(glob(os.path.dirname(cur_dir)+'/RBC_G8_DNS/npdata/'+'*.npy'))
keys = np.arange(len(filenames))
index_ori = np.arange(96*540*540)
for i, filename in enumerate(filenames):
    np.random.seed(keys[i])
    index = np.random.choice(index_ori, size = 25*68*68)
    filenew = np.load(filename).reshape(98,542,542,8)[1:-1,1:-1,1:-1,:].reshape(-1,8)[index,:]
    np.save(os.path.dirname(cur_dir)+'/RBC_G8_DNS/npdata/lv6_xbound/flow0' + str(170+i) + '.npy', filenew)
# %%
cur_dir = os.getcwd()
filenames = sorted(glob(os.path.dirname(cur_dir)+'/RBC_G8_DNS/npdata/lv6_xbound/'+'*.npy'))
file_test = np.load(filenames[2])
#%%
plt.scatter(file_test[:,1], file_test[:,2], s=0.1)
plt.show()
#%%
plt.plot(w_new[:,10,10])
plt.plot(uz[:,10,10])
#plt.plot(u_new[:,11,10])
plt.show()#%%
#%%
print(file_test[:,0])
# %%
filenames = sorted(glob(os.path.dirname(cur_dir)+'/RBC_G8_DNS/npdata/'+'*.npy'))
