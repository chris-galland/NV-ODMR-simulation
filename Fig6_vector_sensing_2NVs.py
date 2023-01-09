# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 17:15:13 2022

@author: goblot
"""

import numpy as np
import matplotlib.pyplot as plt
import simulate_TStrength_NV_ensemble as simnv

# =============================================================================
# Plotting geometrical configuration function
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
def MW_angle_scan(MWfreq, thetak, phik, MWangle, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth):
    # thetak, phik define normal vector for plane of rotation of MW polaritzation
    numa = len(MWangle)
    nMW = len(MWfreq)
    T = np.zeros((numa, nMW))

    e_rot = simnv.get_vector_cartesian(1, thetak, phik) # Rotation vector
    u_rot = simnv.get_vector_cartesian(1, thetak + np.pi/2., phik) # x unitary vector in rotation plane
    v_rot = np.cross(e_rot, u_rot) # y unitary vector in rotation plane
    for idx, ang in enumerate(MWangle):
        mwvec = np.cos(ang) * u_rot + np.sin(ang) * v_rot
        _, thetaMW, phiMW = simnv.get_vector_spherical(mwvec)
        
        T[idx, :] = simnv.ESR_NV_VN_ensemble(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth)
        # T[idx, :] = simnv.ESR_NVensemble(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth)

    return T

# =============================================================================
# Run simulation
# =============================================================================
# Save parameters
sdir = '' # path to directory where figures will be saved
save_flag = False # if True, save figures

# MW frequency
nfreq = 400
freqi = 2850
freqf = 2900

# Linewidth
Linewidth = 1.

# MW propagation direction, in lab frame
nmw = 100 # number of points in polarization angle sweep

thetakmw = 0.
phikmw = 0.

# B in lab frame, orthogonal to NV1 & NV4
thetaB = np.pi/4.
phiB = np.pi/2.

# E in lab frame
nv1 = np.array([1, -1, 1])/ np.sqrt(3)
nv2 = np.array([1, 1, -1])/ np.sqrt(3)
alpha = 1. / np.sqrt(33)
beta = np.sqrt(32 / 33)
e_vec = beta * 3 / np.sqrt(8) * nv2 + (beta / np.sqrt(8) + alpha) * nv1

_, thetaE, phiE = simnv.get_vector_spherical(e_vec)

# Define B and E amplitude
B0 = 20

E0 = 50e6
dE = 1e5

# Plot geometrical configuration
plot_config(thetakmw, phikmw, thetaB, phiB, thetaE, phiE)

# Run simulation and plot results
MWfreq = np.linspace(freqi, freqf, nfreq)
MWangle = np.linspace(0, 2*np.pi, nmw)

TSmap = MW_angle_scan(MWfreq, thetakmw, phikmw, MWangle, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth)


# =============================================================================
# Figures
# =============================================================================
# Intensity map vs Phi_MW
fig_T, ax_T = plt.subplots()
plt.pcolormesh(MWfreq, MWangle / np.pi, TSmap, shading='auto')
plt.colorbar()
plt.xlabel('Frequency (MHz)')
plt.ylabel(r'MW Polarization angle $\phi / \pi$')
plt.ylim(0, 2)
ax_T.set_yticks([0, 0.5, 1., 1.5, 2.])

# Single spectrum, unpolarized MW
fig_o, ax_o = plt.subplots(figsize=(4, 3))
fig_o.subplots_adjust(bottom=0.17, top=0.92, left=0.17, right=0.94)
ax_o.plot(MWfreq, TSmap[0, :])
ax_o.set_xlim(MWfreq.min(), MWfreq.max())
ax_o.set_ylim(ymin=0)
ax_o.set_xlabel('Frequency (MHz)')
ax_o.set_ylabel('Amplitude (arb. units)')
ax_o.set_yticks([0, 0.5, 1., 1.5])

# Intensity at fixed MW frequency vs MW polarization
freq_target = [2862.53, 2866.40] # Low frequency side
# freq_target = [2879.70, 2876.07] # High frequency side

fig_p, ax_p = plt.subplots(figsize=(4, 3))
fig_p.subplots_adjust(bottom=0.17, top=0.92, left=0.17, right=0.94)
for ft in freq_target:
    idx_f = np.argmin(abs(MWfreq - ft))
    ax_p.plot(MWangle / np.pi, TSmap[:, idx_f])
    
    # Optional
    ax_o.axvline(MWfreq[idx_f], ls=':', color='0.7')
ax_p.set_xlim(0, 2)
ax_p.set_ylim(ymin=0)
ax_p.set_xlabel(r'MW Polarization angle $\phi / \pi$')
ax_p.set_ylabel('Amplitude (arb. units)')

if save_flag:
    for fig in [fig_T]:
        fig.set_size_inches(4, 3)
        fig.subplots_adjust(bottom=0.17, top=0.92, left=0.14, right=0.98)
    
    sname = 'Fig6'
    fig_T.savefig(sdir + sname + '_c.png', dpi=200)
    fig_o.savefig(sdir + sname + '_b.png', dpi=200)
    fig_p.savefig(sdir + sname + '_d.png', dpi=200)

