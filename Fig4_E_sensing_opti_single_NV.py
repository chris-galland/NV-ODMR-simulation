# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 15:05:57 2022

@author: goblot
"""

import numpy as np
import matplotlib.pyplot as plt
import simulate_TStrength_NV_ensemble as simnv

# =============================================================================
# Parameter scans
# =============================================================================
def B_scan_single(MWfreq, thetaMW, phiMW, B0list, thetaB, phiB, E0, thetaE, phiE, Linewidth):
    numB =len(B0list)
    nMW = len(MWfreq)
    T = np.zeros((numB, nMW))
    
    # Compute unitary vectors from their polar angles
    MW_vec = simnv.get_vector_cartesian(1, thetaMW, phiMW)
    B_vec = simnv.get_vector_cartesian(1, thetaB, phiB)
    E_vec = simnv.get_vector_cartesian(1, thetaE, phiE)
    
    for idx, B0 in enumerate(B0list):
        T[idx, :] = simnv.ESR_singleNV(MWfreq, MW_vec, B0 * B_vec, E0 * E_vec, Linewidth)
    
    return T
 
def B_scan_single_unpolarizedMW(MWfreq, B0list, thetaB, phiB, E0, thetaE, phiE, Linewidth):
    numB =len(B0list)
    nMW = len(MWfreq)
    T = np.zeros((numB, nMW))
    
    # Compute unitary vectors from their polar angles
    MW_vec_pol1 = simnv.get_vector_cartesian(1, np.pi/2., 0)
    MW_vec_pol2 = simnv.get_vector_cartesian(1, np.pi/2., np.pi/2.)
    B_vec = simnv.get_vector_cartesian(1, thetaB, phiB)
    E_vec = simnv.get_vector_cartesian(1, thetaE, phiE)
    
    for idx, B0 in enumerate(B0list):
        T_pol1 = simnv.ESR_singleNV(MWfreq, MW_vec_pol1, B0 * B_vec, E0 * E_vec, Linewidth)
        T_pol2 = simnv.ESR_singleNV(MWfreq, MW_vec_pol2, B0 * B_vec, E0 * E_vec, Linewidth)
        T[idx, :] = (T_pol1 + T_pol2) / 2.
    
    return T

# =============================================================================
# Run simulation
# =============================================================================
sdir = '' # path to directory where figures will be saved
save_flag = False # if True, save figures

nfreq = 2000
freqi = 2860
freqf = 2890
Linewidth = 1.

# Define E, B and MW directions in NV frame
thetaMW = np.pi/2.
phiMW = np.pi/2.

nB = 100
B0 = 80
thetaB = np.pi/2.
phiB = 0.

E0 = 5e6
thetaE = np.pi/2.
phiE = np.pi/4.

# Run simulation and plot results
MWfreq = np.linspace(freqi, freqf, nfreq)
B0list = np.linspace(0, B0, nB)

# TSmap = B_scan_single(MWfreq, thetaMW, phiMW, B0list, thetaB, phiB, E0, thetaE, phiE, Linewidth)
TSmap = B_scan_single_unpolarizedMW(MWfreq, B0list, thetaB, phiB, E0, thetaE, phiE, Linewidth)

dE = 1e5
# TSmap_dE = B_scan_single(MWfreq, thetaMW, phiMW, B0list, thetaB, phiB, E0 + dE, thetaE, phiE, Linewidth)
TSmap_dE = B_scan_single_unpolarizedMW(MWfreq, B0list, thetaB, phiB, E0 + dE, thetaE, phiE, Linewidth)

# Figures
fig_T, ax_T = plt.subplots()
plt.pcolormesh(MWfreq, B0list, TSmap, shading='auto')
plt.colorbar()
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
plt.xlim(xmin=0)
plt.ylabel('B (G)')
plt.xlabel('Sensitivity (arb. units)')
plt.yticks([0, 20, 40, 60, 80])

for fig in [fig_T, fig_S]:
    fig.set_size_inches(4, 3)
    fig.subplots_adjust(bottom=0.17, top=0.92, left=0.14, right=0.98)
        
if save_flag:
    sname = 'Fig4'
    fig_T.savefig(sdir + sname + '_a.png', dpi=200)
    fig_S.savefig(sdir + sname + '_b.png', dpi=200)
    fig_max.savefig(sdir + sname + '_c.png', dpi=200)
    
