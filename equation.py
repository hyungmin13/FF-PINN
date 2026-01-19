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

class Equation:
    @staticmethod
    def init_params(*args):
        raise NotImplementedError

class FF_equation(Equation):
    def __init__(self, all_params):
        self.all_params = all_params
    
    @staticmethod
    def Loss(dynamic_params, all_params, ff_grid, ff_val, particles, particle_vel, boundaries, model_fns):
        all_params["network"]["layers"] = dynamic_params
        weights = all_params["problem"]["loss_weights1"]

        p_out = model_fns(all_params, particles)
        f_out = model_fns(all_params, ff_grid)

        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)
        loss_fu = all_params["data"]['u_ref']*f_out[:,0:1] - ff_val[:,0:1]
        loss_fu = jnp.mean(loss_fu**2)
        loss_fv = all_params["data"]['v_ref']*f_out[:,1:2] - ff_val[:,1:2]
        loss_fv = jnp.mean(loss_fv**2)
        loss_fw = all_params["data"]['w_ref']*f_out[:,2:3] - ff_val[:,2:3]
        loss_fw = jnp.mean(loss_fw**2)
        loss_fp = all_params["data"]['p_ref']*f_out[:,3:4] - ff_val[:,3:4]
        loss_fp = jnp.mean(loss_fp**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_fu + weights[4]*loss_fv + weights[5]*loss_fw + \
                    weights[6]*loss_fp
        return total_loss
    
    @staticmethod
    def Loss_report(dynamic_params, all_params, ff_grid, ff_val, particles, particle_vel, boundaries, model_fns):
        all_params["network"]["layers"] = dynamic_params
        weights = all_params["problem"]["loss_weights1"]

        p_out = model_fns(all_params, particles)                                                                        
        f_out = model_fns(all_params, ff_grid)
        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_fu = all_params["data"]['u_ref']*f_out[:,0:1] - ff_val[:,0:1]
        loss_fu = jnp.mean(loss_fu**2)
        loss_fv = all_params["data"]['v_ref']*f_out[:,1:2] - ff_val[:,1:2]
        loss_fv = jnp.mean(loss_fv**2)
        loss_fw = all_params["data"]['w_ref']*f_out[:,2:3] - ff_val[:,2:3]
        loss_fw = jnp.mean(loss_fw**2)
        loss_fp = all_params["data"]['p_ref']*f_out[:,3:4] - ff_val[:,3:4]
        loss_fp = jnp.mean(loss_fp**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_fu + weights[4]*loss_fv + weights[5]*loss_fw + \
                    weights[6]*loss_fp
        return total_loss, loss_u, loss_v, loss_w, loss_fu, loss_fv, loss_fw, loss_fp

class Boundless_flow_FF(Equation):
    def __init__(self, all_params):
        self.all_params = all_params
    
    @staticmethod
    def Loss(dynamic_params, all_params, loss_factor, g_batch, ff_grid, ff_val, particles, particle_vel, boundaries, model_fns):
        def first_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
            return out, out_t

        def second_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            def u_tt(batch):
                return jax.jvp(u_t,(batch,), (cotangent, ))[1]
            out_x, out_xx = jax.jvp(u_tt, (g_batch,), (cotangent,))
            return out_x, out_xx
        all_params["network"]["layers"] = dynamic_params
        weights = all_params["problem"]["loss_weights2"]
        out, out_t = first_order(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_x, out_xx = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_y, out_yy = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_z, out_zz = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)

        p_out = model_fns(all_params, particles)                                                                           
        f_out = model_fns(all_params, ff_grid)
        u = all_params["data"]['u_ref']*out[:,0:1]
        v = all_params["data"]['v_ref']*out[:,1:2]
        w = all_params["data"]['w_ref']*out[:,2:3]
        p = all_params["data"]['p_ref']*out[:,3:4]

        ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["domain"]["domain_range"]["t"][1]
        vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["domain"]["domain_range"]["t"][1]
        wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["domain"]["domain_range"]["t"][1]

        ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["domain"]["domain_range"]["x"][1]
        vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["domain"]["domain_range"]["x"][1]
        wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["domain"]["domain_range"]["x"][1]
        px = all_params["data"]['p_ref']*out_x[:,3:4]/all_params["domain"]["domain_range"]["x"][1]

        uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["domain"]["domain_range"]["y"][1]
        vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["domain"]["domain_range"]["y"][1]
        wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["domain"]["domain_range"]["y"][1]
        py = all_params["data"]['p_ref']*out_y[:,3:4]/all_params["domain"]["domain_range"]["y"][1]

        uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["domain"]["domain_range"]["z"][1]
        vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["domain"]["domain_range"]["z"][1]
        wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["domain"]["domain_range"]["z"][1]
        pz = all_params["data"]['p_ref']*out_z[:,3:4]/all_params["domain"]["domain_range"]["z"][1]

        uxx = all_params["data"]['u_ref']*out_xx[:,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
        vxx = all_params["data"]['v_ref']*out_xx[:,1:2]/all_params["domain"]["domain_range"]["x"][1]**2
        wxx = all_params["data"]['w_ref']*out_xx[:,2:3]/all_params["domain"]["domain_range"]["x"][1]**2

        uyy = all_params["data"]['u_ref']*out_yy[:,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
        vyy = all_params["data"]['v_ref']*out_yy[:,1:2]/all_params["domain"]["domain_range"]["y"][1]**2
        wyy = all_params["data"]['w_ref']*out_yy[:,2:3]/all_params["domain"]["domain_range"]["y"][1]**2

        uzz = all_params["data"]['u_ref']*out_zz[:,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
        vzz = all_params["data"]['v_ref']*out_zz[:,1:2]/all_params["domain"]["domain_range"]["z"][1]**2
        wzz = all_params["data"]['w_ref']*out_zz[:,2:3]/all_params["domain"]["domain_range"]["z"][1]**2

        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_fu = all_params["data"]['u_ref']*f_out[:,0:1] - ff_val[:,0:1]
        loss_fu = jnp.mean(loss_fu**2)
        loss_fv = all_params["data"]['v_ref']*f_out[:,1:2] - ff_val[:,1:2]
        loss_fv = jnp.mean(loss_fv**2)
        loss_fw = all_params["data"]['w_ref']*f_out[:,2:3] - ff_val[:,2:3]
        loss_fw = jnp.mean(loss_fw**2)
        loss_fp = all_params["data"]['p_ref']*f_out[:,3:4] - ff_val[:,3:4]
        loss_fp = jnp.mean(loss_fp**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)-2.22*10**(-1)/(3*0.43685**2)*u
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)-2.22*10**(-1)/(3*0.43685**2)*v
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz)-2.22*10**(-1)/(3*0.43685**2)*w
        loss_NS3 = jnp.mean(loss_NS3**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + weights[7]*loss_con + weights[8]*loss_NS1 + weights[9]*loss_NS2 + weights[10]*loss_NS3
                    #loss_factor*(weights[3]*loss_fu + weights[4]*loss_fv + weights[5]*loss_fw) + weights[6]*loss_fp + \
                    #(1-loss_factor)*(weights[7]*loss_con + weights[8]*loss_NS1 + weights[9]*loss_NS2 + weights[10]*loss_NS3)
        return total_loss
    
    @staticmethod
    def Loss_report(dynamic_params, all_params, loss_factor, g_batch, ff_grid, ff_val, particles, particle_vel, boundaries, model_fns):
        def first_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
            return out, out_t

        def second_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            def u_tt(batch):
                return jax.jvp(u_t,(batch,), (cotangent, ))[1]
            out_x, out_xx = jax.jvp(u_tt, (g_batch,), (cotangent,))
            return out_x, out_xx
        all_params["network"]["layers"] = dynamic_params
        weights = all_params["problem"]["loss_weights2"]
        out, out_t = first_order(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_x, out_xx = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_y, out_yy = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_z, out_zz = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)

        p_out = model_fns(all_params, particles)                                                                        
        f_out = model_fns(all_params, ff_grid)
        u = all_params["data"]['u_ref']*out[:,0:1]
        v = all_params["data"]['v_ref']*out[:,1:2]
        w = all_params["data"]['w_ref']*out[:,2:3]
        p = all_params["data"]['p_ref']*out[:,3:4]

        ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["domain"]["domain_range"]["t"][1]
        vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["domain"]["domain_range"]["t"][1]
        wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["domain"]["domain_range"]["t"][1]

        ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["domain"]["domain_range"]["x"][1]
        vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["domain"]["domain_range"]["x"][1]
        wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["domain"]["domain_range"]["x"][1]
        px = all_params["data"]['p_ref']*out_x[:,3:4]/all_params["domain"]["domain_range"]["x"][1]

        uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["domain"]["domain_range"]["y"][1]
        vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["domain"]["domain_range"]["y"][1]
        wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["domain"]["domain_range"]["y"][1]
        py = all_params["data"]['p_ref']*out_y[:,3:4]/all_params["domain"]["domain_range"]["y"][1]

        uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["domain"]["domain_range"]["z"][1]
        vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["domain"]["domain_range"]["z"][1]
        wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["domain"]["domain_range"]["z"][1]
        pz = all_params["data"]['p_ref']*out_z[:,3:4]/all_params["domain"]["domain_range"]["z"][1]

        uxx = all_params["data"]['u_ref']*out_xx[:,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
        vxx = all_params["data"]['v_ref']*out_xx[:,1:2]/all_params["domain"]["domain_range"]["x"][1]**2
        wxx = all_params["data"]['w_ref']*out_xx[:,2:3]/all_params["domain"]["domain_range"]["x"][1]**2

        uyy = all_params["data"]['u_ref']*out_yy[:,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
        vyy = all_params["data"]['v_ref']*out_yy[:,1:2]/all_params["domain"]["domain_range"]["y"][1]**2
        wyy = all_params["data"]['w_ref']*out_yy[:,2:3]/all_params["domain"]["domain_range"]["y"][1]**2

        uzz = all_params["data"]['u_ref']*out_zz[:,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
        vzz = all_params["data"]['v_ref']*out_zz[:,1:2]/all_params["domain"]["domain_range"]["z"][1]**2
        wzz = all_params["data"]['w_ref']*out_zz[:,2:3]/all_params["domain"]["domain_range"]["z"][1]**2

        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_fu = all_params["data"]['u_ref']*f_out[:,0:1] - ff_val[:,0:1]
        loss_fu = jnp.mean(loss_fu**2)
        loss_fv = all_params["data"]['v_ref']*f_out[:,1:2] - ff_val[:,1:2]
        loss_fv = jnp.mean(loss_fv**2)
        loss_fw = all_params["data"]['w_ref']*f_out[:,2:3] - ff_val[:,2:3]
        loss_fw = jnp.mean(loss_fw**2)
        loss_fp = all_params["data"]['p_ref']*f_out[:,3:4] - ff_val[:,3:4]
        loss_fp = jnp.mean(loss_fp**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)-2.22*10**(-1)/(3*0.43685**2)*u
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)-2.22*10**(-1)/(3*0.43685**2)*v
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz)-2.22*10**(-1)/(3*0.43685**2)*w
        loss_NS3 = jnp.mean(loss_NS3**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + weights[7]*loss_con + weights[8]*loss_NS1 + weights[9]*loss_NS2 + weights[10]*loss_NS3
                    #loss_factor*(weights[3]*loss_fu + weights[4]*loss_fv + weights[5]*loss_fw + weights[6]*loss_fp) + \
                    #(1-loss_factor)*(weights[7]*loss_con + weights[8]*loss_NS1 + weights[9]*loss_NS2 + weights[10]*loss_NS3)
        return total_loss, loss_u, loss_v, loss_w, loss_fu, loss_fv, loss_fw, loss_fp, loss_con, loss_NS1, loss_NS2, loss_NS3


class Boundless_flow(Equation):
    def __init__(self, all_params):
        self.all_params = all_params
    
    @staticmethod
    def Loss(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
        def first_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
            return out, out_t

        def second_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            def u_tt(batch):
                return jax.jvp(u_t,(batch,), (cotangent, ))[1]
            out_x, out_xx = jax.jvp(u_tt, (g_batch,), (cotangent,))
            return out_x, out_xx
        all_params["network"]["layers"] = dynamic_params
        weights = all_params["problem"]["loss_weights"]
        out, out_t = first_order(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_x, out_xx = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_y, out_yy = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_z, out_zz = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)

        p_out = model_fns(all_params, particles)                                                                           

        u = all_params["data"]['u_ref']*out[:,0:1]
        v = all_params["data"]['v_ref']*out[:,1:2]
        w = all_params["data"]['w_ref']*out[:,2:3]
        p = all_params["data"]['p_ref']*out[:,3:4]

        ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["domain"]["domain_range"]["t"][1]
        vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["domain"]["domain_range"]["t"][1]
        wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["domain"]["domain_range"]["t"][1]

        ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["domain"]["domain_range"]["x"][1]
        vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["domain"]["domain_range"]["x"][1]
        wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["domain"]["domain_range"]["x"][1]
        px = all_params["data"]['p_ref']*out_x[:,3:4]/all_params["domain"]["domain_range"]["x"][1]

        uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["domain"]["domain_range"]["y"][1]
        vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["domain"]["domain_range"]["y"][1]
        wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["domain"]["domain_range"]["y"][1]
        py = all_params["data"]['p_ref']*out_y[:,3:4]/all_params["domain"]["domain_range"]["y"][1]

        uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["domain"]["domain_range"]["z"][1]
        vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["domain"]["domain_range"]["z"][1]
        wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["domain"]["domain_range"]["z"][1]
        pz = all_params["data"]['p_ref']*out_z[:,3:4]/all_params["domain"]["domain_range"]["z"][1]

        uxx = all_params["data"]['u_ref']*out_xx[:,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
        vxx = all_params["data"]['v_ref']*out_xx[:,1:2]/all_params["domain"]["domain_range"]["x"][1]**2
        wxx = all_params["data"]['w_ref']*out_xx[:,2:3]/all_params["domain"]["domain_range"]["x"][1]**2

        uyy = all_params["data"]['u_ref']*out_yy[:,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
        vyy = all_params["data"]['v_ref']*out_yy[:,1:2]/all_params["domain"]["domain_range"]["y"][1]**2
        wyy = all_params["data"]['w_ref']*out_yy[:,2:3]/all_params["domain"]["domain_range"]["y"][1]**2

        uzz = all_params["data"]['u_ref']*out_zz[:,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
        vzz = all_params["data"]['v_ref']*out_zz[:,1:2]/all_params["domain"]["domain_range"]["z"][1]**2
        wzz = all_params["data"]['w_ref']*out_zz[:,2:3]/all_params["domain"]["domain_range"]["z"][1]**2

        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)-2.22*10**(-1)/(3*0.43685**2)*u
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)-2.22*10**(-1)/(3*0.43685**2)*v
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz)-2.22*10**(-1)/(3*0.43685**2)*w
        loss_NS3 = jnp.mean(loss_NS3**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 
        return total_loss
    
    @staticmethod
    def Loss_report(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
        def first_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
            return out, out_t

        def second_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            def u_tt(batch):
                return jax.jvp(u_t,(batch,), (cotangent, ))[1]
            out_x, out_xx = jax.jvp(u_tt, (g_batch,), (cotangent,))
            return out_x, out_xx
        all_params["network"]["layers"] = dynamic_params
        weights = all_params["problem"]["loss_weights"]
        out, out_t = first_order(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_x, out_xx = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_y, out_yy = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_z, out_zz = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)

        p_out = model_fns(all_params, particles)                                                                        

        u = all_params["data"]['u_ref']*out[:,0:1]
        v = all_params["data"]['v_ref']*out[:,1:2]
        w = all_params["data"]['w_ref']*out[:,2:3]
        p = all_params["data"]['p_ref']*out[:,3:4]

        ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["domain"]["domain_range"]["t"][1]
        vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["domain"]["domain_range"]["t"][1]
        wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["domain"]["domain_range"]["t"][1]

        ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["domain"]["domain_range"]["x"][1]
        vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["domain"]["domain_range"]["x"][1]
        wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["domain"]["domain_range"]["x"][1]
        px = all_params["data"]['p_ref']*out_x[:,3:4]/all_params["domain"]["domain_range"]["x"][1]


        uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["domain"]["domain_range"]["y"][1]
        vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["domain"]["domain_range"]["y"][1]
        wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["domain"]["domain_range"]["y"][1]
        py = all_params["data"]['p_ref']*out_y[:,3:4]/all_params["domain"]["domain_range"]["y"][1]

        uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["domain"]["domain_range"]["z"][1]
        vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["domain"]["domain_range"]["z"][1]
        wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["domain"]["domain_range"]["z"][1]
        pz = all_params["data"]['p_ref']*out_z[:,3:4]/all_params["domain"]["domain_range"]["z"][1]

        uxx = all_params["data"]['u_ref']*out_xx[:,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
        vxx = all_params["data"]['v_ref']*out_xx[:,1:2]/all_params["domain"]["domain_range"]["x"][1]**2
        wxx = all_params["data"]['w_ref']*out_xx[:,2:3]/all_params["domain"]["domain_range"]["x"][1]**2

        uyy = all_params["data"]['u_ref']*out_yy[:,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
        vyy = all_params["data"]['v_ref']*out_yy[:,1:2]/all_params["domain"]["domain_range"]["y"][1]**2
        wyy = all_params["data"]['w_ref']*out_yy[:,2:3]/all_params["domain"]["domain_range"]["y"][1]**2

        uzz = all_params["data"]['u_ref']*out_zz[:,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
        vzz = all_params["data"]['v_ref']*out_zz[:,1:2]/all_params["domain"]["domain_range"]["z"][1]**2
        wzz = all_params["data"]['w_ref']*out_zz[:,2:3]/all_params["domain"]["domain_range"]["z"][1]**2

        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)-2.22*10**(-1)/(3*0.43685**2)*u
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)-2.22*10**(-1)/(3*0.43685**2)*v
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz)-2.22*10**(-1)/(3*0.43685**2)*w
        loss_NS3 = jnp.mean(loss_NS3**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 
        return total_loss, loss_u, loss_v, loss_w, loss_con, loss_NS1, loss_NS2, loss_NS3

class Boundary_layer(Equation):
    def __init__(self, all_params):
        self.all_params = all_params
    
    @staticmethod
    def Loss(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
        def first_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
            return out, out_t

        def second_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            def u_tt(batch):
                return jax.jvp(u_t,(batch,), (cotangent, ))[1]
            out_x, out_xx = jax.jvp(u_tt, (g_batch,), (cotangent,))
            return out_x, out_xx
        all_params["network"]["layers"] = dynamic_params
        weights = all_params["problem"]["loss_weights"]
        out, out_t = first_order(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_x, out_xx = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_y, out_yy = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_z, out_zz = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)

        p_out = model_fns(all_params, particles)                                                                           
        b_out = model_fns(all_params, boundaries[0])
        u = all_params["data"]['u_ref']*out[:,0:1]
        v = all_params["data"]['v_ref']*out[:,1:2]
        w = all_params["data"]['w_ref']*out[:,2:3]
        p = all_params["data"]['p_ref']*out[:,3:4]

        ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["domain"]["domain_range"]["t"][1]
        vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["domain"]["domain_range"]["t"][1]
        wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["domain"]["domain_range"]["t"][1]

        ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["domain"]["domain_range"]["x"][1]
        vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["domain"]["domain_range"]["x"][1]
        wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["domain"]["domain_range"]["x"][1]
        px = all_params["data"]['p_ref']*out_x[:,3:4]/all_params["domain"]["domain_range"]["x"][1]

        uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["domain"]["domain_range"]["y"][1]
        vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["domain"]["domain_range"]["y"][1]
        wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["domain"]["domain_range"]["y"][1]
        py = all_params["data"]['p_ref']*out_y[:,3:4]/all_params["domain"]["domain_range"]["y"][1]

        uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["domain"]["domain_range"]["z"][1]
        vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["domain"]["domain_range"]["z"][1]
        wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["domain"]["domain_range"]["z"][1]
        pz = all_params["data"]['p_ref']*out_z[:,3:4]/all_params["domain"]["domain_range"]["z"][1]

        uxx = all_params["data"]['u_ref']*out_xx[:,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
        vxx = all_params["data"]['v_ref']*out_xx[:,1:2]/all_params["domain"]["domain_range"]["x"][1]**2
        wxx = all_params["data"]['w_ref']*out_xx[:,2:3]/all_params["domain"]["domain_range"]["x"][1]**2

        uyy = all_params["data"]['u_ref']*out_yy[:,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
        vyy = all_params["data"]['v_ref']*out_yy[:,1:2]/all_params["domain"]["domain_range"]["y"][1]**2
        wyy = all_params["data"]['w_ref']*out_yy[:,2:3]/all_params["domain"]["domain_range"]["y"][1]**2

        uzz = all_params["data"]['u_ref']*out_zz[:,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
        vzz = all_params["data"]['v_ref']*out_zz[:,1:2]/all_params["domain"]["domain_range"]["z"][1]**2
        wzz = all_params["data"]['w_ref']*out_zz[:,2:3]/all_params["domain"]["domain_range"]["z"][1]**2

        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_u_b = all_params["data"]['u_ref']*b_out[:,0:1]
        loss_u_b = jnp.mean(loss_u_b**2)

        loss_v_b = all_params["data"]['v_ref']*b_out[:,1:2]
        loss_v_b = jnp.mean(loss_v_b**2)

        loss_w_b = all_params["data"]['w_ref']*b_out[:,2:3]
        loss_w_b = jnp.mean(loss_w_b**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz)
        loss_NS3 = jnp.mean(loss_NS3**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 + weights[7]*(loss_u_b + loss_v_b + loss_w_b)
        return total_loss
    @staticmethod
    def Loss_report(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
        def first_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
            return out, out_t

        def second_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            def u_tt(batch):
                return jax.jvp(u_t,(batch,), (cotangent, ))[1]
            out_x, out_xx = jax.jvp(u_tt, (g_batch,), (cotangent,))
            return out_x, out_xx
        all_params["network"]["layers"] = dynamic_params
        weights = all_params["problem"]["loss_weights"]
        out, out_t = first_order(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_x, out_xx = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_y, out_yy = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_z, out_zz = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)

        p_out = model_fns(all_params, particles)                                                                        
        b_out = model_fns(all_params, boundaries[0])

        u = all_params["data"]['u_ref']*out[:,0:1]
        v = all_params["data"]['v_ref']*out[:,1:2]
        w = all_params["data"]['w_ref']*out[:,2:3]
        p = all_params["data"]['p_ref']*out[:,3:4]

        ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["domain"]["domain_range"]["t"][1]
        vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["domain"]["domain_range"]["t"][1]
        wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["domain"]["domain_range"]["t"][1]

        ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["domain"]["domain_range"]["x"][1]
        vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["domain"]["domain_range"]["x"][1]
        wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["domain"]["domain_range"]["x"][1]
        px = all_params["data"]['p_ref']*out_x[:,3:4]/all_params["domain"]["domain_range"]["x"][1]


        uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["domain"]["domain_range"]["y"][1]
        vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["domain"]["domain_range"]["y"][1]
        wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["domain"]["domain_range"]["y"][1]
        py = all_params["data"]['p_ref']*out_y[:,3:4]/all_params["domain"]["domain_range"]["y"][1]

        uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["domain"]["domain_range"]["z"][1]
        vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["domain"]["domain_range"]["z"][1]
        wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["domain"]["domain_range"]["z"][1]
        pz = all_params["data"]['p_ref']*out_z[:,3:4]/all_params["domain"]["domain_range"]["z"][1]

        uxx = all_params["data"]['u_ref']*out_xx[:,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
        vxx = all_params["data"]['v_ref']*out_xx[:,1:2]/all_params["domain"]["domain_range"]["x"][1]**2
        wxx = all_params["data"]['w_ref']*out_xx[:,2:3]/all_params["domain"]["domain_range"]["x"][1]**2

        uyy = all_params["data"]['u_ref']*out_yy[:,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
        vyy = all_params["data"]['v_ref']*out_yy[:,1:2]/all_params["domain"]["domain_range"]["y"][1]**2
        wyy = all_params["data"]['w_ref']*out_yy[:,2:3]/all_params["domain"]["domain_range"]["y"][1]**2

        uzz = all_params["data"]['u_ref']*out_zz[:,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
        vzz = all_params["data"]['v_ref']*out_zz[:,1:2]/all_params["domain"]["domain_range"]["z"][1]**2
        wzz = all_params["data"]['w_ref']*out_zz[:,2:3]/all_params["domain"]["domain_range"]["z"][1]**2

        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_u_b = all_params["data"]['u_ref']*b_out[:,0:1]
        loss_u_b = jnp.mean(loss_u_b**2)

        loss_v_b = all_params["data"]['v_ref']*b_out[:,1:2]
        loss_v_b = jnp.mean(loss_v_b**2)

        loss_w_b = all_params["data"]['w_ref']*b_out[:,2:3]
        loss_w_b = jnp.mean(loss_w_b**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz)
        loss_NS3 = jnp.mean(loss_NS3**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 + weights[7]*(loss_u_b + loss_v_b + loss_w_b)
        return total_loss, loss_u, loss_v, loss_w, loss_con, loss_NS1, loss_NS2, loss_NS3

class DUCT_flow(Equation):
    def __init__(self, all_params):
        self.all_params = all_params
    
    @staticmethod
    def Loss(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
        def first_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
            return out, out_t

        def second_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            def u_tt(batch):
                return jax.jvp(u_t,(batch,), (cotangent, ))[1]
            out_x, out_xx = jax.jvp(u_tt, (g_batch,), (cotangent,))
            return out_x, out_xx
        all_params["network"]["layers"] = dynamic_params
        weights = all_params["problem"]["loss_weights"]
        out, out_t = first_order(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_x, out_xx = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_y, out_yy = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_z, out_zz = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)

        p_out = model_fns(all_params, particles)                                                                           
        b_out1 = model_fns(all_params, boundaries[0])
        b_out2 = model_fns(all_params, boundaries[1])
        u = all_params["data"]['u_ref']*out[:,0:1]
        v = all_params["data"]['v_ref']*out[:,1:2]
        w = all_params["data"]['w_ref']*out[:,2:3]
        p = all_params["data"]['p_ref']*out[:,3:4]

        ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["domain"]["domain_range"]["t"][1]
        vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["domain"]["domain_range"]["t"][1]
        wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["domain"]["domain_range"]["t"][1]

        ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["domain"]["domain_range"]["x"][1]
        vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["domain"]["domain_range"]["x"][1]
        wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["domain"]["domain_range"]["x"][1]
        px = all_params["data"]['p_ref']*out_x[:,3:4]/all_params["domain"]["domain_range"]["x"][1]

        uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["domain"]["domain_range"]["y"][1]
        vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["domain"]["domain_range"]["y"][1]
        wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["domain"]["domain_range"]["y"][1]
        py = all_params["data"]['p_ref']*out_y[:,3:4]/all_params["domain"]["domain_range"]["y"][1]

        uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["domain"]["domain_range"]["z"][1]
        vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["domain"]["domain_range"]["z"][1]
        wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["domain"]["domain_range"]["z"][1]
        pz = all_params["data"]['p_ref']*out_z[:,3:4]/all_params["domain"]["domain_range"]["z"][1]

        uxx = all_params["data"]['u_ref']*out_xx[:,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
        vxx = all_params["data"]['v_ref']*out_xx[:,1:2]/all_params["domain"]["domain_range"]["x"][1]**2
        wxx = all_params["data"]['w_ref']*out_xx[:,2:3]/all_params["domain"]["domain_range"]["x"][1]**2

        uyy = all_params["data"]['u_ref']*out_yy[:,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
        vyy = all_params["data"]['v_ref']*out_yy[:,1:2]/all_params["domain"]["domain_range"]["y"][1]**2
        wyy = all_params["data"]['w_ref']*out_yy[:,2:3]/all_params["domain"]["domain_range"]["y"][1]**2

        uzz = all_params["data"]['u_ref']*out_zz[:,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
        vzz = all_params["data"]['v_ref']*out_zz[:,1:2]/all_params["domain"]["domain_range"]["z"][1]**2
        wzz = all_params["data"]['w_ref']*out_zz[:,2:3]/all_params["domain"]["domain_range"]["z"][1]**2

        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_u_b1 = all_params["data"]['u_ref']*b_out1[:,0:1]
        loss_u_b1 = jnp.mean(loss_u_b1**2)

        loss_v_b1 = all_params["data"]['v_ref']*b_out1[:,1:2]
        loss_v_b1 = jnp.mean(loss_v_b1**2)

        loss_w_b1 = all_params["data"]['w_ref']*b_out1[:,2:3]
        loss_w_b1 = jnp.mean(loss_w_b1**2)

        loss_u_b2 = all_params["data"]['u_ref']*b_out2[:,0:1]
        loss_u_b2 = jnp.mean(loss_u_b2**2)

        loss_v_b2 = all_params["data"]['v_ref']*b_out2[:,1:2]
        loss_v_b2 = jnp.mean(loss_v_b2**2)

        loss_w_b2 = all_params["data"]['w_ref']*b_out2[:,2:3]
        loss_w_b2 = jnp.mean(loss_w_b2**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz)
        loss_NS3 = jnp.mean(loss_NS3**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 + weights[7]*(loss_u_b1 + loss_v_b1 + loss_w_b1 + loss_u_b2 + loss_v_b2 + loss_w_b2)
        return total_loss
    @staticmethod
    def Loss_report(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
        def first_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
            return out, out_t

        def second_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            def u_tt(batch):
                return jax.jvp(u_t,(batch,), (cotangent, ))[1]
            out_x, out_xx = jax.jvp(u_tt, (g_batch,), (cotangent,))
            return out_x, out_xx
        all_params["network"]["layers"] = dynamic_params
        weights = all_params["problem"]["loss_weights"]
        out, out_t = first_order(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_x, out_xx = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_y, out_yy = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_z, out_zz = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)

        p_out = model_fns(all_params, particles)                                                                        
        b_out1 = model_fns(all_params, boundaries[0])
        b_out2 = model_fns(all_params, boundaries[1])
        
        u = all_params["data"]['u_ref']*out[:,0:1]
        v = all_params["data"]['v_ref']*out[:,1:2]
        w = all_params["data"]['w_ref']*out[:,2:3]
        p = all_params["data"]['p_ref']*out[:,3:4]

        ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["domain"]["domain_range"]["t"][1]
        vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["domain"]["domain_range"]["t"][1]
        wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["domain"]["domain_range"]["t"][1]

        ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["domain"]["domain_range"]["x"][1]
        vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["domain"]["domain_range"]["x"][1]
        wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["domain"]["domain_range"]["x"][1]
        px = all_params["data"]['p_ref']*out_x[:,3:4]/all_params["domain"]["domain_range"]["x"][1]


        uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["domain"]["domain_range"]["y"][1]
        vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["domain"]["domain_range"]["y"][1]
        wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["domain"]["domain_range"]["y"][1]
        py = all_params["data"]['p_ref']*out_y[:,3:4]/all_params["domain"]["domain_range"]["y"][1]

        uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["domain"]["domain_range"]["z"][1]
        vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["domain"]["domain_range"]["z"][1]
        wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["domain"]["domain_range"]["z"][1]
        pz = all_params["data"]['p_ref']*out_z[:,3:4]/all_params["domain"]["domain_range"]["z"][1]

        uxx = all_params["data"]['u_ref']*out_xx[:,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
        vxx = all_params["data"]['v_ref']*out_xx[:,1:2]/all_params["domain"]["domain_range"]["x"][1]**2
        wxx = all_params["data"]['w_ref']*out_xx[:,2:3]/all_params["domain"]["domain_range"]["x"][1]**2

        uyy = all_params["data"]['u_ref']*out_yy[:,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
        vyy = all_params["data"]['v_ref']*out_yy[:,1:2]/all_params["domain"]["domain_range"]["y"][1]**2
        wyy = all_params["data"]['w_ref']*out_yy[:,2:3]/all_params["domain"]["domain_range"]["y"][1]**2

        uzz = all_params["data"]['u_ref']*out_zz[:,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
        vzz = all_params["data"]['v_ref']*out_zz[:,1:2]/all_params["domain"]["domain_range"]["z"][1]**2
        wzz = all_params["data"]['w_ref']*out_zz[:,2:3]/all_params["domain"]["domain_range"]["z"][1]**2

        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_u_b1 = all_params["data"]['u_ref']*b_out1[:,0:1]
        loss_u_b1 = jnp.mean(loss_u_b1**2)

        loss_v_b1 = all_params["data"]['v_ref']*b_out1[:,1:2]
        loss_v_b1 = jnp.mean(loss_v_b1**2)

        loss_w_b1 = all_params["data"]['w_ref']*b_out1[:,2:3]
        loss_w_b1 = jnp.mean(loss_w_b1**2)

        loss_u_b2 = all_params["data"]['u_ref']*b_out2[:,0:1]
        loss_u_b2 = jnp.mean(loss_u_b2**2)

        loss_v_b2 = all_params["data"]['v_ref']*b_out2[:,1:2]
        loss_v_b2 = jnp.mean(loss_v_b2**2)

        loss_w_b2 = all_params["data"]['w_ref']*b_out2[:,2:3]
        loss_w_b2 = jnp.mean(loss_w_b2**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz)
        loss_NS3 = jnp.mean(loss_NS3**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 + weights[7]*(loss_u_b1 + loss_v_b1 + loss_w_b1 + loss_u_b2 + loss_v_b2 + loss_w_b2)
        return total_loss, loss_u, loss_v, loss_w, loss_con, loss_NS1, loss_NS2, loss_NS3

class Energy(Equation):
    def __init__(self, all_params):
        self.all_params = all_params
    
    @staticmethod
    def Loss(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
        def first_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
            return out, out_t

        def second_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            def u_tt(batch):
                return jax.jvp(u_t,(batch,), (cotangent, ))[1]
            out_x, out_xx = jax.jvp(u_tt, (g_batch,), (cotangent,))
            return out_x, out_xx
   
        all_params["network"]["layers"] = dynamic_params
        weights = all_params["problem"]["loss_weights"]
        out, out_t = first_order(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_x, out_xx = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_y, out_yy = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_z, out_zz = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)

        p_out = model_fns(all_params, particles)
        b_out1 = model_fns(all_params, boundaries[0])                                                                                  
        b_out2 = model_fns(all_params, boundaries[1])                                                                                  

        u = all_params["data"]['u_ref']*out[:,0:1]
        v = all_params["data"]['v_ref']*out[:,1:2]
        w = all_params["data"]['w_ref']*out[:,2:3]
        p = all_params["data"]['p_ref']*out[:,3:4]
        T = all_params["data"]['T_ref']*out[:,4:5]

        ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["domain"]["domain_range"]["t"][1]
        vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["domain"]["domain_range"]["t"][1]
        wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["domain"]["domain_range"]["t"][1]
        Tt = all_params["data"]['T_ref']*out_t[:,4:5]/all_params["domain"]["domain_range"]["t"][1]

        ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["domain"]["domain_range"]["x"][1]
        vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["domain"]["domain_range"]["x"][1]
        wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["domain"]["domain_range"]["x"][1]
        px = all_params["data"]['p_ref']*out_x[:,3:4]/all_params["domain"]["domain_range"]["x"][1]
        Tx = all_params["data"]['T_ref']*out_x[:,4:5]/all_params["domain"]["domain_range"]["x"][1]

        uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["domain"]["domain_range"]["y"][1]
        vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["domain"]["domain_range"]["y"][1]
        wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["domain"]["domain_range"]["y"][1]
        py = all_params["data"]['p_ref']*out_y[:,3:4]/all_params["domain"]["domain_range"]["y"][1]
        Ty = all_params["data"]['T_ref']*out_y[:,4:5]/all_params["domain"]["domain_range"]["y"][1]

        uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["domain"]["domain_range"]["z"][1]
        vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["domain"]["domain_range"]["z"][1]
        wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["domain"]["domain_range"]["z"][1]
        pz = all_params["data"]['p_ref']*out_z[:,3:4]/all_params["domain"]["domain_range"]["z"][1]
        Tz = all_params["data"]['T_ref']*out_z[:,4:5]/all_params["domain"]["domain_range"]["z"][1]

        uxx = all_params["data"]['u_ref']*out_xx[:,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
        vxx = all_params["data"]['v_ref']*out_xx[:,1:2]/all_params["domain"]["domain_range"]["x"][1]**2
        wxx = all_params["data"]['w_ref']*out_xx[:,2:3]/all_params["domain"]["domain_range"]["x"][1]**2
        Txx = all_params["data"]['T_ref']*out_xx[:,4:5]/all_params["domain"]["domain_range"]["x"][1]**2

        uyy = all_params["data"]['u_ref']*out_yy[:,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
        vyy = all_params["data"]['v_ref']*out_yy[:,1:2]/all_params["domain"]["domain_range"]["y"][1]**2
        wyy = all_params["data"]['w_ref']*out_yy[:,2:3]/all_params["domain"]["domain_range"]["y"][1]**2
        Tyy = all_params["data"]['T_ref']*out_yy[:,4:5]/all_params["domain"]["domain_range"]["y"][1]**2

        uzz = all_params["data"]['u_ref']*out_zz[:,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
        vzz = all_params["data"]['v_ref']*out_zz[:,1:2]/all_params["domain"]["domain_range"]["z"][1]**2
        wzz = all_params["data"]['w_ref']*out_zz[:,2:3]/all_params["domain"]["domain_range"]["z"][1]**2
        Tzz = all_params["data"]['T_ref']*out_zz[:,4:5]/all_params["domain"]["domain_range"]["z"][1]**2

        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_T_bu = all_params["data"]['T_ref']*b_out1[:,4:5] + all_params["data"]['T_ref']
        loss_T_bu = jnp.mean(loss_T_bu**2)
        
        loss_T_bb = all_params["data"]['T_ref']*b_out2[:,4:5] - all_params["data"]['T_ref']
        loss_T_bb = jnp.mean(loss_T_bb**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz) - T

        loss_NS3 = jnp.mean(loss_NS3**2)

        loss_ENR = Tt + u*Tx + v*Ty + w*Tz - all_params["data"]["viscosity"]*(Txx+Tyy+Tzz)/7
        loss_ENR = jnp.mean(loss_ENR**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 + weights[7]*loss_ENR + weights[8]*(loss_T_bu + loss_T_bb)
        return total_loss
    @staticmethod
    def Loss_report(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
        def first_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
            return out, out_t

        def second_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            def u_tt(batch):
                return jax.jvp(u_t,(batch,), (cotangent, ))[1]
            out_x, out_xx = jax.jvp(u_tt, (g_batch,), (cotangent,))
            return out_x, out_xx
        all_params["network"]["layers"] = dynamic_params
        weights = all_params["problem"]["loss_weights"]
        out, out_t = first_order(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_x, out_xx = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_y, out_yy = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_z, out_zz = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)

        p_out = model_fns(all_params, particles)
        b_out1 = model_fns(all_params, boundaries[0])                                                                                  
        b_out2 = model_fns(all_params, boundaries[1])                                                                                  

        u = all_params["data"]['u_ref']*out[:,0:1]
        v = all_params["data"]['v_ref']*out[:,1:2]
        w = all_params["data"]['w_ref']*out[:,2:3]
        p = all_params["data"]['p_ref']*out[:,3:4]
        T = all_params["data"]['T_ref']*out[:,4:5]
        ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["domain"]["domain_range"]["t"][1]
        vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["domain"]["domain_range"]["t"][1]
        wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["domain"]["domain_range"]["t"][1]
        Tt = all_params["data"]['T_ref']*out_t[:,4:5]/all_params["domain"]["domain_range"]["t"][1]

        ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["domain"]["domain_range"]["x"][1]
        vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["domain"]["domain_range"]["x"][1]
        wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["domain"]["domain_range"]["x"][1]
        px = all_params["data"]['p_ref']*out_x[:,3:4]/all_params["domain"]["domain_range"]["x"][1]
        Tx = all_params["data"]['T_ref']*out_x[:,4:5]/all_params["domain"]["domain_range"]["x"][1]

        uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["domain"]["domain_range"]["y"][1]
        vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["domain"]["domain_range"]["y"][1]
        wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["domain"]["domain_range"]["y"][1]
        py = all_params["data"]['p_ref']*out_y[:,3:4]/all_params["domain"]["domain_range"]["y"][1]
        Ty = all_params["data"]['T_ref']*out_y[:,4:5]/all_params["domain"]["domain_range"]["y"][1]

        uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["domain"]["domain_range"]["z"][1]
        vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["domain"]["domain_range"]["z"][1]
        wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["domain"]["domain_range"]["z"][1]
        pz = all_params["data"]['p_ref']*out_z[:,3:4]/all_params["domain"]["domain_range"]["z"][1]
        Tz = all_params["data"]['T_ref']*out_z[:,4:5]/all_params["domain"]["domain_range"]["z"][1]

        uxx = all_params["data"]['u_ref']*out_xx[:,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
        vxx = all_params["data"]['v_ref']*out_xx[:,1:2]/all_params["domain"]["domain_range"]["x"][1]**2
        wxx = all_params["data"]['w_ref']*out_xx[:,2:3]/all_params["domain"]["domain_range"]["x"][1]**2
        Txx = all_params["data"]['T_ref']*out_xx[:,4:5]/all_params["domain"]["domain_range"]["x"][1]**2

        uyy = all_params["data"]['u_ref']*out_yy[:,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
        vyy = all_params["data"]['v_ref']*out_yy[:,1:2]/all_params["domain"]["domain_range"]["y"][1]**2
        wyy = all_params["data"]['w_ref']*out_yy[:,2:3]/all_params["domain"]["domain_range"]["y"][1]**2
        Tyy = all_params["data"]['T_ref']*out_yy[:,4:5]/all_params["domain"]["domain_range"]["y"][1]**2

        uzz = all_params["data"]['u_ref']*out_zz[:,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
        vzz = all_params["data"]['v_ref']*out_zz[:,1:2]/all_params["domain"]["domain_range"]["z"][1]**2
        wzz = all_params["data"]['w_ref']*out_zz[:,2:3]/all_params["domain"]["domain_range"]["z"][1]**2
        Tzz = all_params["data"]['T_ref']*out_zz[:,4:5]/all_params["domain"]["domain_range"]["z"][1]**2

        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_T_bu = all_params["data"]['T_ref']*b_out1[:,4:5] + all_params["data"]['T_ref']
        loss_T_bu = jnp.mean(loss_T_bu**2)
        
        loss_T_bb = all_params["data"]['T_ref']*b_out2[:,4:5] - all_params["data"]['T_ref']
        loss_T_bb = jnp.mean(loss_T_bb**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz) - T

        loss_NS3 = jnp.mean(loss_NS3**2)

        loss_ENR = Tt + u*Tx + v*Ty + w*Tz - all_params["data"]["viscosity"]*(Txx+Tyy+Tzz)/7
        loss_ENR = jnp.mean(loss_ENR**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 + weights[7]*loss_ENR + weights[8]*(loss_T_bu + loss_T_bb)
        return total_loss, loss_u, loss_v, loss_w, loss_con, loss_NS1, loss_NS2, loss_NS3, loss_ENR, loss_T_bu, loss_T_bb
class Energy_hard_profile(Equation):
    def __init__(self, all_params):
        self.all_params = all_params
    
    @staticmethod
    def Loss(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
        def first_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
            return out, out_t

        def second_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            def u_tt(batch):
                return jax.jvp(u_t,(batch,), (cotangent, ))[1]
            out_x, out_xx = jax.jvp(u_tt, (g_batch,), (cotangent,))
            return out_x, out_xx
        
        def temperature_profile(z):
            denom = 2-2*jnp.exp(-0.5*30)
            return jnp.where(z > 0.5,
                            (jnp.exp(-z * 30) - jnp.exp(-0.5 * 30)) / denom,
                            (jnp.exp(-0.5 * 30) - jnp.exp(30 * (z - 1))) / denom
                            )
        def temperature_profile_z(z):
            denom = 2-2*jnp.exp(-0.5*30)
            return jnp.where(z > 0.5,
                            -30*jnp.exp(-z * 30) / denom,
                            -30*jnp.exp(30 * (z - 1)) / denom
                            )
        def temperature_profile_zz(z):
            denom = 2-2*jnp.exp(-0.5*30)
            return jnp.where(z > 0.5,
                            900*jnp.exp(-z * 30) / denom,
                            -900*jnp.exp(30 * (z - 1)) / denom
                            )
   
        all_params["network"]["layers"] = dynamic_params
        weights = all_params["problem"]["loss_weights"]
        out, out_t = first_order(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_x, out_xx = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_y, out_yy = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_z, out_zz = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)

        p_out = model_fns(all_params, particles)
        b_out1 = model_fns(all_params, boundaries[0])                                                                                  
        b_out2 = model_fns(all_params, boundaries[1])                                                                                  
        temp_mean = temperature_profile(g_batch[:,3:4])
        temp_mean_z = temperature_profile_z(g_batch[:,3:4])
        temp_mean_zz = temperature_profile_zz(g_batch[:,3:4])
        u = all_params["data"]['u_ref']*out[:,0:1]
        v = all_params["data"]['v_ref']*out[:,1:2]
        w = all_params["data"]['w_ref']*out[:,2:3]
        p = all_params["data"]['p_ref']*out[:,3:4]
        T_ = all_params["data"]['T_ref']*out[:,4:5]
        T = (g_batch[:,3:4]**2-g_batch[:,3:4])*T_ + temp_mean
        ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["domain"]["domain_range"]["t"][1]
        vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["domain"]["domain_range"]["t"][1]
        wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["domain"]["domain_range"]["t"][1]
        Tt = (g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_t[:,4:5]/all_params["domain"]["domain_range"]["t"][1]

        ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["domain"]["domain_range"]["x"][1]
        vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["domain"]["domain_range"]["x"][1]
        wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["domain"]["domain_range"]["x"][1]
        px = all_params["data"]['p_ref']*out_x[:,3:4]/all_params["domain"]["domain_range"]["x"][1]
        Tx = (g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_x[:,4:5]/all_params["domain"]["domain_range"]["x"][1]

        uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["domain"]["domain_range"]["y"][1]
        vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["domain"]["domain_range"]["y"][1]
        wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["domain"]["domain_range"]["y"][1]
        py = all_params["data"]['p_ref']*out_y[:,3:4]/all_params["domain"]["domain_range"]["y"][1]
        Ty = (g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_y[:,4:5]/all_params["domain"]["domain_range"]["y"][1]

        uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["domain"]["domain_range"]["z"][1]
        vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["domain"]["domain_range"]["z"][1]
        wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["domain"]["domain_range"]["z"][1]
        pz = all_params["data"]['p_ref']*out_z[:,3:4]/all_params["domain"]["domain_range"]["z"][1]
        Tz = (2*g_batch[:,3:4]-1)*T_+(g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_z[:,4:5]/all_params["domain"]["domain_range"]["z"][1] + temp_mean_z

        uxx = all_params["data"]['u_ref']*out_xx[:,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
        vxx = all_params["data"]['v_ref']*out_xx[:,1:2]/all_params["domain"]["domain_range"]["x"][1]**2
        wxx = all_params["data"]['w_ref']*out_xx[:,2:3]/all_params["domain"]["domain_range"]["x"][1]**2
        Txx = (g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_xx[:,4:5]/all_params["domain"]["domain_range"]["x"][1]**2

        uyy = all_params["data"]['u_ref']*out_yy[:,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
        vyy = all_params["data"]['v_ref']*out_yy[:,1:2]/all_params["domain"]["domain_range"]["y"][1]**2
        wyy = all_params["data"]['w_ref']*out_yy[:,2:3]/all_params["domain"]["domain_range"]["y"][1]**2
        Tyy = (g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_yy[:,4:5]/all_params["domain"]["domain_range"]["y"][1]**2

        uzz = all_params["data"]['u_ref']*out_zz[:,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
        vzz = all_params["data"]['v_ref']*out_zz[:,1:2]/all_params["domain"]["domain_range"]["z"][1]**2
        wzz = all_params["data"]['w_ref']*out_zz[:,2:3]/all_params["domain"]["domain_range"]["z"][1]**2
        Tzz = 2*T_+(4*g_batch[:,3:4]-2)*all_params["data"]['T_ref']*out_z[:,4:5]/all_params["domain"]["domain_range"]["z"][1]\
              +(g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_zz[:,4:5]/all_params["domain"]["domain_range"]["z"][1]**2 + temp_mean_zz

        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_T_bu = all_params["data"]['T_ref']*b_out1[:,4:5] + all_params["data"]['T_ref']
        loss_T_bu = jnp.mean(loss_T_bu**2)
        
        loss_T_bb = all_params["data"]['T_ref']*b_out2[:,4:5] - all_params["data"]['T_ref']
        loss_T_bb = jnp.mean(loss_T_bb**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz) - T

        loss_NS3 = jnp.mean(loss_NS3**2)

        loss_ENR = Tt + u*Tx + v*Ty + w*Tz - all_params["data"]["viscosity"]*(Txx+Tyy+Tzz)/7
        loss_ENR = jnp.mean(loss_ENR**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 + weights[7]*loss_ENR + weights[8]*(loss_T_bu + loss_T_bb)
        return total_loss
    @staticmethod
    def Loss_report(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
        def first_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
            return out, out_t

        def second_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            def u_tt(batch):
                return jax.jvp(u_t,(batch,), (cotangent, ))[1]
            out_x, out_xx = jax.jvp(u_tt, (g_batch,), (cotangent,))
            return out_x, out_xx
        def temperature_profile(z):
            denom = 2-2*jnp.exp(-0.5*30)
            return jnp.where(z > 0.5,
                            (jnp.exp(-z * 30) - jnp.exp(-0.5 * 30)) / denom,
                            (jnp.exp(-0.5 * 30) - jnp.exp(30 * (z - 1))) / denom
                            )
        def temperature_profile_z(z):
            denom = 2-2*jnp.exp(-0.5*30)
            return jnp.where(z > 0.5,
                            -30*jnp.exp(-z * 30) / denom,
                            -30*jnp.exp(30 * (z - 1)) / denom
                            )
        def temperature_profile_zz(z):
            denom = 2-2*jnp.exp(-0.5*30)
            return jnp.where(z > 0.5,
                            900*jnp.exp(-z * 30) / denom,
                            -900*jnp.exp(30 * (z - 1)) / denom
                            )
        all_params["network"]["layers"] = dynamic_params
        weights = all_params["problem"]["loss_weights"]
        out, out_t = first_order(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_x, out_xx = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_y, out_yy = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_z, out_zz = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)

        p_out = model_fns(all_params, particles)
        b_out1 = model_fns(all_params, boundaries[0])                                                                                  
        b_out2 = model_fns(all_params, boundaries[1])                                                                                  
        temp_mean = temperature_profile(g_batch[:,3:4])
        temp_mean_z = temperature_profile_z(g_batch[:,3:4])
        temp_mean_zz = temperature_profile_zz(g_batch[:,3:4])
        u = all_params["data"]['u_ref']*out[:,0:1]
        v = all_params["data"]['v_ref']*out[:,1:2]
        w = all_params["data"]['w_ref']*out[:,2:3]
        p = all_params["data"]['p_ref']*out[:,3:4]
        T_ = all_params["data"]['T_ref']*out[:,4:5]
        T = (g_batch[:,3:4]**2-g_batch[:,3:4])*T_ + temp_mean
        ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["domain"]["domain_range"]["t"][1]
        vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["domain"]["domain_range"]["t"][1]
        wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["domain"]["domain_range"]["t"][1]
        Tt = (g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_t[:,4:5]/all_params["domain"]["domain_range"]["t"][1]

        ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["domain"]["domain_range"]["x"][1]
        vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["domain"]["domain_range"]["x"][1]
        wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["domain"]["domain_range"]["x"][1]
        px = all_params["data"]['p_ref']*out_x[:,3:4]/all_params["domain"]["domain_range"]["x"][1]
        Tx = (g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_x[:,4:5]/all_params["domain"]["domain_range"]["x"][1]

        uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["domain"]["domain_range"]["y"][1]
        vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["domain"]["domain_range"]["y"][1]
        wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["domain"]["domain_range"]["y"][1]
        py = all_params["data"]['p_ref']*out_y[:,3:4]/all_params["domain"]["domain_range"]["y"][1]
        Ty = (g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_y[:,4:5]/all_params["domain"]["domain_range"]["y"][1]

        uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["domain"]["domain_range"]["z"][1]
        vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["domain"]["domain_range"]["z"][1]
        wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["domain"]["domain_range"]["z"][1]
        pz = all_params["data"]['p_ref']*out_z[:,3:4]/all_params["domain"]["domain_range"]["z"][1]
        Tz = (2*g_batch[:,3:4]-1)*T_+(g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_z[:,4:5]/all_params["domain"]["domain_range"]["z"][1] + temp_mean_z

        uxx = all_params["data"]['u_ref']*out_xx[:,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
        vxx = all_params["data"]['v_ref']*out_xx[:,1:2]/all_params["domain"]["domain_range"]["x"][1]**2
        wxx = all_params["data"]['w_ref']*out_xx[:,2:3]/all_params["domain"]["domain_range"]["x"][1]**2
        Txx = (g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_xx[:,4:5]/all_params["domain"]["domain_range"]["x"][1]**2

        uyy = all_params["data"]['u_ref']*out_yy[:,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
        vyy = all_params["data"]['v_ref']*out_yy[:,1:2]/all_params["domain"]["domain_range"]["y"][1]**2
        wyy = all_params["data"]['w_ref']*out_yy[:,2:3]/all_params["domain"]["domain_range"]["y"][1]**2
        Tyy = (g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_yy[:,4:5]/all_params["domain"]["domain_range"]["y"][1]**2

        uzz = all_params["data"]['u_ref']*out_zz[:,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
        vzz = all_params["data"]['v_ref']*out_zz[:,1:2]/all_params["domain"]["domain_range"]["z"][1]**2
        wzz = all_params["data"]['w_ref']*out_zz[:,2:3]/all_params["domain"]["domain_range"]["z"][1]**2
        Tzz = 2*T_+(4*g_batch[:,3:4]-2)*all_params["data"]['T_ref']*out_z[:,4:5]/all_params["domain"]["domain_range"]["z"][1]\
              +(g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_zz[:,4:5]/all_params["domain"]["domain_range"]["z"][1]**2 + temp_mean_zz


        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_T_bu = all_params["data"]['T_ref']*b_out1[:,4:5] + all_params["data"]['T_ref']
        loss_T_bu = jnp.mean(loss_T_bu**2)
        
        loss_T_bb = all_params["data"]['T_ref']*b_out2[:,4:5] - all_params["data"]['T_ref']
        loss_T_bb = jnp.mean(loss_T_bb**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz) - T

        loss_NS3 = jnp.mean(loss_NS3**2)

        loss_ENR = Tt + u*Tx + v*Ty + w*Tz - all_params["data"]["viscosity"]*(Txx+Tyy+Tzz)/7
        loss_ENR = jnp.mean(loss_ENR**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 + weights[7]*loss_ENR + weights[8]*(loss_T_bu + loss_T_bb)
        return total_loss, loss_u, loss_v, loss_w, loss_con, loss_NS1, loss_NS2, loss_NS3, loss_ENR, loss_T_bu, loss_T_bb
