### CDS for turbulence in 3D
import numpy as np
from numba import jit, prange
import timeit
import copy
from scipy import signal
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
import scipy.ndimage
import torch
import torch.nn as nn
import h5py

## Simulation setup
dx = 0.05
L = 5.
x = np.arange(-L, L + dx, dx)
y = np.arange(-L, L + dx, dx)
z = np.arange(-L, L + dx, dx)
Nx = len(x)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
R = np.sqrt(X**2 + Y**2 + Z**2)
r = x[(Nx//2):]
Nr = len(r)

# Device for PyTorch
device = (
    #torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # For Nvidia GPU
    #torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  # For Apple GPU
    torch.device('cpu')  # Default to CPU
)
print(device)

# find the average q(r) over the solid angle through binning
def average_qr(q_xyz, X, Y, Z, r):
    R = np.sqrt(X**2 + Y**2 + Z**2)
    qr = np.zeros(len(r))
    dr = r[1] - r[0];
    r_max = max(r)
    
    for i in range(len(r)):
        r_l = max(r[i] - dr/2, 0)
        r_u = min(r[i] + dr/2, r_max)
        in_bin = (R >= r_l) & (R < r_u)  # Points that fall within the bin
        qr[i] = np.mean(q_xyz[in_bin])  # Average density for this bin
        
    return qr
    
def average2d_qr(q_xyz, Y, Z, r):
    # Assuming the middle index corresponds to x = 0
    Nx = q_xyz.shape[0]
    mid_index = Nx // 2  # Integer division to get the middle index
    
    # Take a 2D slice at x = 0
    q_slice = q_xyz[mid_index, :, :]
    
    # Calculate radial distances in the y-z plane
    R = np.sqrt(Y[mid_index, :, :]**2 + Z[mid_index, :, :]**2)
    
    qr = np.zeros(len(r))
    dr = r[1] - r[0]
    r_max = max(r)
    
    for i in range(len(r)):
        r_l = max(r[i] - dr/2, 0)
        r_u = min(r[i] + dr/2, r_max)
        in_bin = (R >= r_l) & (R < r_u)  # Points that fall within the bin
        
        # Weighted average for this bin, weighting by r to account for the Jacobian
        if np.any(in_bin):
            qr[i] = np.mean(q_slice[in_bin] * R[in_bin]) / np.mean(R[in_bin])
        else:
            qr[i] = np.nan  # Avoid division by zero if no points fall within the bin
        
    return qr
    

# Convolution kernels as NumPy arrays
laplc_3d_SO = 40/11 * np.array([[[1/80, 3/80, 1/80],
                                    [3/80, 6/80, 3/80],
                                    [1/80, 3/80, 1/80]],
                                   [[3/80, 6/80, 3/80],
                                    [6/80, -1., 6/80],
                                    [3/80, 6/80, 3/80]],
                                   [[1/80, 3/80, 1/80],
                                    [3/80, 6/80, 3/80],
                                    [1/80, 3/80, 1/80]]])

laplc_3d_d3q27 = 1/36 * np.array([[[1, 4, 1],
                                      [4, 16, 4],
                                      [1, 4, 1]],
                                     [[4, 16, 4],
                                      [16, -152, 16],
                                      [4, 16, 4]],
                                     [[1, 4, 1],
                                      [4, 16, 4],
                                      [1, 4, 1]]])

laplc_3d_d3q15 = 1/12 * np.array([[[1, 0, 1],
                                      [0, 8, 0],
                                      [1, 0, 1]],
                                     [[0, 8, 0],
                                      [8, -56, 8],
                                      [0, 8, 0]],
                                     [[1, 0, 1],
                                      [0, 8, 0],
                                      [1, 0, 1]]])

#@jit(nopython=True)
def pad_matrix(matrix, boundary_condition='replicate', rem_rate=1):
    if boundary_condition == 'zero':
        return np.pad(matrix, 1, 'constant', constant_values=0)
    elif boundary_condition == 'replicate':
        return np.pad(matrix, 1, 'edge')
    elif boundary_condition == 'wrap':
        return np.pad(matrix, 1, 'wrap')
    elif boundary_condition == 'absorption':
        padded_matrix = np.pad(matrix, 1, 'edge')
        # Scale the padding by the remaining rate
        padded_matrix[0, :, :] *= rem_rate
        padded_matrix[-1, :, :] *= rem_rate
        padded_matrix[:, 0, :] *= rem_rate
        padded_matrix[:, -1, :] *= rem_rate
        padded_matrix[:, :, 0] *= rem_rate
        padded_matrix[:, :, -1] *= rem_rate
        return padded_matrix
    else:
        raise ValueError("Unsupported boundary condition")

@jit(nopython=True, parallel=True)
def my_convolve3d(padded_matrix, kernel):
    x, y, z = padded_matrix.shape
    x -= 2  # Adjusting for padding
    y -= 2
    z -= 2
    output = np.zeros((x, y, z))

    for i in prange(x):
        for j in range(y):
            for k in range(z):
                output[i, j, k] = np.sum(kernel * padded_matrix[i:i+3, j:j+3, k:k+3])

    return output

## Update the equation using CDS & using ell
# Step 1: Onsite dynamics for q
@jit(nopython=True)
def onsite_dynamics_2(q1, ell, dt, epsilon):
    q2 = q1 * (1 + epsilon * dt * np.sqrt(q1) / (2 * ell)) ** (-2)
    return q2

# Step 2: Coupled map using CDS for q, isotropic Laplacian in 3D
#@jit(nopython=True)
def coupled_3d_2(q1, ell, dt, c, rem_rate):
    qn = np.power(q1, 3/2)
    laplc_qn = (
        my_convolve3d(pad_matrix(qn, boundary_condition='absorption', rem_rate=rem_rate), laplc_3d_d3q27)
    )
    dq2 = (2/3) * c * ell * dt * laplc_qn / (dx**2)
    return dq2

# Complete update
#@jit(nopython=True)
def turbulence_3d_cds_2(q1, ell, dt, c, rem_rate, epsilon):
    # Step 1: Onsite dynamics for q
    q2 = onsite_dynamics_2(q1, ell, dt, epsilon)
    # Step 2: Coupled map using CDS for q
    q3 = q2 + coupled_3d_2(q1, ell, dt, c, rem_rate)
    return q3
    
    
r3d = np.arange(0, L * np.sqrt(3), dx)
r2d = np.arange(0, L * np.sqrt(2), dx)

### Func to run an Entire simulation
def sim_run(q3d0, c=0.5, rem_rate=0.8, epsilon=1.0, dt=0.001, t_start=0., t_end=0.1, saveFilename='test.h5', saveInterval=100, model_name='simulation_bl_cds'):
    t = np.arange(t_start, t_end + dt, dt)
    Nt = len(t)
    
    q3d = q3d0
    qr = average_qr(q3d0, X, Y, Z, r3d)
    qr2d = average2d_qr(q3d0, Y, Z, r2d)
    ell = ell_sim(t)

    with h5py.File(saveFilename, 'w') as f:
        # Save the model name as an attribute of the root group
        f.attrs['model_name'] = model_name
        
        # Create a group for simulation parameters and save parameters there
        params_group = f.create_group('parameters')
        params_group.create_dataset('c', data=c)
        params_group.create_dataset('epsilon', data=epsilon)
        params_group.create_dataset('rem_rate', data=rem_rate)
        params_group.create_dataset('saveInterval', data=saveInterval)
        params_group.create_dataset('dt', data=dt)
        params_group.create_dataset('t_start', data=t_start)
        params_group.create_dataset('t_end', data=t_end)
        params_group.create_dataset('Nt', data=Nt)
        params_group.create_dataset('L', data=L)
        
        # Save spatial and time coordinates
        f.create_dataset('t', data=t)
        f.create_dataset('r', data=r)
        f.create_dataset('r2d', data=r2d)
        f.create_dataset('r3d', data=r3d)

        f.create_dataset('qr_timestep0', data=qr)
        f.create_dataset('qr2d_timestep0', data=qr2d)
        f.create_dataset('ell', data=ell)
        f.create_dataset('q3d_start', data=q3d0)
        
        start_time = timeit.default_timer()
        
        for i in range(Nt-1):
            q3d = turbulence_3d_cds_2(q3d, ell[i], dt, c, rem_rate, epsilon)
     
            if (i+1)%saveInterval == 0:
                print(i + 1)
                # Save simulation results
                qr = average_qr(q3d, X, Y, Z, r3d)
                f.create_dataset(f'qr_timestep{i+1}', data=qr)
                
        end_time = timeit.default_timer()
        print(f"Time taken for the loop: {end_time - start_time:.2f} seconds")

        f.create_dataset('q3d_end', data=q3d)
        print(f'Data saved to {saveFilename}')
    
    return q3d, qr, qr2d
    
    
## Set initial condition
# import initial condition from experiment
with h5py.File('smallBlob/IC_smallBlob_t1p85.h5', 'r') as f:
    q3d0 = f['q3d0'][()]
    L_e = f['L_e'][()]
    t0_e = f['t0_e'][()]
    tauq_e = f['tauq_e'][()]
    tauq_s = f['tauq_s'][()]
    
## Prescribe integral length scale ell(t) (do not change)
# import integral length scale from experiment
with h5py.File('smallBlob/ell_e_filtered.h5', 'r') as f:
    t_ell_e = f['t_ell_e'][:] # in seconds
    ell_filtered_e = f['ell_filtered_e'][:]# Integral length scale in mm

# interpolate to find simulation ell
def ell_sim(t_s):
    t_e = t_s * tauq_e/tauq_s + t0_e
    ell_intr_e = np.interp(t_e, t_ell_e, ell_filtered_e)
    ell_s = ell_intr_e * L/L_e
    return ell_s

## Set parameters
c_0 = 1.2
epsilon_0 = 0.88
rem_rate = 0.9


### Run!
q3d_end, qr_end, qr2d_end = sim_run(q3d0, c=c_0, rem_rate=rem_rate, epsilon=epsilon_0, dt=0.0005, t_start=0., t_end=1., saveFilename=f'smallBlob_t1p85_intrEll_bl_sim_0_1.h5', saveInterval=20, model_name='simulation_bl_cds')
