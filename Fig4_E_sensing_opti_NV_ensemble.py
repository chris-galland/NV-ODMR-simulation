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
def B_scan(MWfreq, thetaMW, phiMW, B0list, thetaB, phiB, E0, thetaE, phiE, Linewidth):
    numB =len(B0list)
    nMW = len(MWfreq)
    T = np.zeros((numB, nMW))
    
    for idx, B0 in enumerate(B0list):
        T[idx, :] = simnv.ESR_NV_VN_ensemble(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth)
    
    return T

def B_scan_unpolarizedMW(MWfreq, thetak, phik, B0list, thetaB, phiB, E0, thetaE, phiE, Linewidth):
    numB =len(B0list)
    nMW = len(MWfreq)
    T = np.zeros((numB, nMW))
    
    for idx, B0 in enumerate(B0list):
        T_pol1 = simnv.ESR_NV_VN_ensemble(MWfreq, thetak + np.pi/2., phik, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth)
        T_pol2 = simnv.ESR_NV_VN_ensemble(MWfreq, np.pi/2., phik + np.pi/2., B0, thetaB, phiB, E0, thetaE, phiE, Linewidth)
        T[idx, :] = (T_pol1 + T_pol2) / 2.
    
    return T
    
# =============================================================================
# Run simulation
# =============================================================================
# Save parameters
sdir = '' # path to directory where figures will be saved
save_flag = False # if True, save figures

# MW frequency
nfreq = 2000
freqi = 2860
freqf = 2890

# Linewidth
Linewidth = 1

# Define MW directions in lab frame
thetaMW = np.pi/2.
phiMW = np.pi/4.

# Define E, B directions in NV1 frame
thetaB = np.pi/2.
phiB = 0.

thetaE = np.pi/2
phiE = np.pi/4.

# Define B and E amplitude
nB = 300
B0 = 80

E0 = 5e6
dE = 1e5

# Compute E, B directions in lab frame
thetaB, phiB = simnv.transform_spherical_nv_to_lab_frame(thetaB, phiB)
thetaE, phiE = simnv.transform_spherical_nv_to_lab_frame(thetaE, phiE)

# Plot geometrical configuration
plot_config(thetaMW, phiMW, thetaB, phiB, thetaE, phiE)

# Run simulation and plot results
MWfreq = np.linspace(freqi, freqf, nfreq)
B0list = np.linspace(0, B0, nB)

TSmap = B_scan(MWfreq, thetaMW, phiMW, B0list, thetaB, phiB, E0, thetaE, phiE, Linewidth)

TSmap_dE = B_scan(MWfreq, thetaMW, phiMW, B0list, thetaB, phiB, E0 + dE, thetaE, phiE, Linewidth)

# Figures
fig_T, ax_T = plt.subplots()
im = plt.pcolormesh(MWfreq, B0list, TSmap, shading='auto')
plt.colorbar(im)
plt.xlabel('Frequency (MHz)')
plt.ylabel('B (G)')

fig_S, ax_S = plt.subplots()
im = plt.pcolormesh(MWfreq, B0list, TSmap - TSmap_dE, shading='auto', cmap='RdGy')
plt.colorbar(im)
vmax = max(np.max(TSmap - TSmap_dE), abs(np.min(TSmap - TSmap_dE)))
im.set_clim(-vmax, vmax)
plt.xlabel('Frequency (MHz)')
plt.ylabel('B (G)')

fig_max, ax_max = plt.subplots(figsize=(2.2, 3))
plt.subplots_adjust(bottom=0.17, top=0.92, left=0.25, right=0.94)
plt.plot(np.max(TSmap - TSmap_dE, axis=-1), B0list)
plt.plot(-np.min(TSmap - TSmap_dE, axis=-1), B0list, color='C0', ls='--')
plt.ylim(5, B0)
plt.xlim(xmin=0, xmax=0.0595)
plt.ylabel('B (G)')
plt.xlabel('Sensitivity (arb. units)')
plt.xticks([0, 0.02, 0.04, 0.04])
plt.yticks([0, 20, 40, 60, 80])

for fig in [fig_T, fig_S]:
    fig.set_size_inches(4, 3)
    fig.subplots_adjust(bottom=0.17, top=0.92, left=0.14, right=0.95)

if save_flag:
    sname = 'Fig4'
    fig_T.savefig(sdir + sname + '_d.png', dpi=200)
    fig_S.savefig(sdir + sname + '_e.png', dpi=200)
    fig_max.savefig(sdir + sname + '_f.png', dpi=200)
