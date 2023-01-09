# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 17:15:13 2022

@author: goblot
"""

import numpy as np
import matplotlib.pyplot as plt
import simulate_TStrength_NV_ensemble as simnv

# =============================================================================
# Plotting geometricl configuration function
# =============================================================================
def plot_vector(vec, ax, length=1, color='k'):
    arrow = 0.06 / length
    ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], 
              length=length, arrow_length_ratio=arrow, color=color)

def plot_config(thetaMW, phiMW, thetaB, phiB, thetaE, phiE):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    
    # add 4 NV orientations
    plot_vector([1, -1, 1], ax) # NV1
    plot_vector([-1, 1, 1], ax) # NV2
    plot_vector([-1, -1, -1], ax) # NV3
    plot_vector([1, 1, -1], ax) # NV4

    # Plot MW, B and E
    plot_vector(simnv.get_vector_cartesian(1, thetaB, phiB), ax, color='C0')
    plot_vector(simnv.get_vector_cartesian(1., thetaE, phiE), ax, color='C1', length=2/3.)
    plot_vector(simnv.get_vector_cartesian(1., thetaMW, phiMW), ax, color='C2', length=1/3.)

# =============================================================================
# Parameter scans
# =============================================================================
def Bangle_scan(MWfreq, thetaMW, phiMW, B0, thetaBlist, phiB, E0, thetaE, phiE, Linewidth):
    numB =len(thetaBlist)
    nMW = len(MWfreq)
    T = np.zeros((numB, nMW))
    
    for idx, thetaB in enumerate(thetaBlist):
        T[idx, :] = simnv.ESR_NV_VN_ensemble(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth)
    
    return T

    
# =============================================================================
# Run simulation
# =============================================================================
# Save parameters
sdir = '' # path to directory where figures will be saved
save_flag = False # if True, save figures

# MW frequency
nfreq = 500
freqi = 2870
freqf = 2900

# Linewidth
Linewidth = 1.

# Define MW directions in lab frame
thetaMW = np.pi/2.
phiMW = np.pi/4.

# Define E, B directions in NV1 frame
thetaB = np.pi/2.
phiB = 0.

thetaE = np.pi/2
phiE = np.pi/4.

# Define B and E amplitude
B0 = 60

E0 = 5e6
dE = 1e5

# Define angle sweep
ntheta = 301
thetam = 5 # misalignement angle in degrees
dthetalist = np.linspace(-thetam, thetam, ntheta)

# Transform E and B coordinates to lab frame
thetaB, phiB = simnv.transform_spherical_nv_to_lab_frame(thetaB, phiB)
thetaE, phiE = simnv.transform_spherical_nv_to_lab_frame(thetaE, phiE)

# Plot geometrical configuration
plot_config(thetaMW, phiMW, thetaB, phiB, thetaE, phiE)

# Run simulation and plot results
MWfreq = np.linspace(freqi, freqf, nfreq)

thetalist = thetaB + dthetalist * np.pi / 180

TSmap = Bangle_scan(MWfreq, thetaMW, phiMW, B0, thetalist, phiB, E0, thetaE, phiE, Linewidth)

TSmap_dE = Bangle_scan(MWfreq, thetaMW, phiMW, B0, thetalist, phiB, E0 + dE, thetaE, phiE, Linewidth)


# Figures
fig_T, ax_T = plt.subplots()
plt.pcolormesh(MWfreq, dthetalist, TSmap, shading='auto')
plt.colorbar()
plt.xlabel('Frequency (MHz)')
plt.ylabel('$\delta\\theta_B$ (°)', labelpad=-1)

fig_S, ax_S = plt.subplots()
im = plt.pcolormesh(MWfreq, dthetalist, TSmap - TSmap_dE, shading='auto', cmap='RdGy')
plt.colorbar(im)
vmax = max(np.max(TSmap - TSmap_dE), abs(np.min(TSmap - TSmap_dE)))
im.set_clim(-vmax, vmax)
plt.xlabel('Frequency (MHz)')
plt.ylabel('$\delta\\theta_B$ (°)', labelpad=-1)

fig_max, ax_max = plt.subplots(figsize=(2.2, 3))
plt.subplots_adjust(bottom=0.17, top=0.92, left=0.25, right=0.94)
plt.plot(np.max(TSmap - TSmap_dE, axis=-1), dthetalist)
plt.plot(-np.min(TSmap - TSmap_dE, axis=-1), dthetalist, color='C0', ls='--')
plt.ylim(-thetam, thetam)
plt.xlim(xmin=0)
plt.ylabel('$\delta\\theta_B$ (°)', labelpad=-1)
plt.xlabel('Sensitivity (arb. units)')

for fig in [fig_T, fig_S]:
    fig.set_size_inches(4, 3)
    fig.subplots_adjust(bottom=0.17, top=0.92, left=0.14, right=0.95)

if save_flag:
    sname = 'Fig5'
    fig_T.savefig(sdir + sname + '_a.png', dpi=200)
    fig_S.savefig(sdir + sname + '_b.png', dpi=200)
    fig_max.savefig(sdir + sname + '_c.png', dpi=200)

