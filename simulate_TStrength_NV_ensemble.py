# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 15:58:11 2022

@author: goblot
"""

import numpy as np

# This script considers the 4 NV orientations in lab frame are
# NV1: [ 1 -1  1]
# NV2: [-1  1  1]
# NV3: [-1 -1 -1]
# NV4: [ 1  1 -1]

# =============================================================================
# Tranformation of vector coordinates
# ============================================================================= 
# Generic functions
def get_vector_cartesian(A, theta, phi):
    """ 
    Compute cartesian coordinates of a vector from its spherical coordinates:
    norm A, polar angle theta, azimutal angle phi
    """
    vec = np.array([A * np.sin(theta) * np.cos(phi), 
                    A * np.sin(theta) * np.sin(phi),
                    A * np.cos(theta)])
    return vec

def get_vector_spherical(Avec):
    """ Compute spherical coordinates of a vector fron its cartesian
    coordinates """
    A0 = np.sqrt(np.dot(Avec, Avec))
    theta = np.arccos(Avec[2] / A0)
    try:
        phi = np.arctan(Avec[1] / Avec[0])
    except ZeroDivisionError:
        phi = 0.
    if np.isnan(phi):
        phi = 0.
    if Avec[0] < 0:
        phi += np.pi
    return A0, theta, phi

# Transformation between lab frame and NV frames, cartesian coordinates
def get_rotation_matrix(idx_nv):
    """ Returns the transformation matrix from lab frame to the desired 
    NV frame, identified by idx_nv (can be 1, 2, 3 or 4) """
    if idx_nv==1:
        RNV = np.array([[1/np.sqrt(6), -1/np.sqrt(6), -2/np.sqrt(6)],
                        [1/np.sqrt(2),  1/np.sqrt(2),  0],
                        [1/np.sqrt(3), -1/np.sqrt(3),  1/np.sqrt(3)]])
    elif idx_nv==2:
        RNV = np.array([[-1/np.sqrt(6),  1/np.sqrt(6), -2/np.sqrt(6)],
                        [-1/np.sqrt(2), -1/np.sqrt(2),  0],
                        [-1/np.sqrt(3),  1/np.sqrt(3),  1/np.sqrt(3)]])
    elif idx_nv==3:
        RNV = np.array([[-1/np.sqrt(6), -1/np.sqrt(6),  2/np.sqrt(6)],
                        [-1/np.sqrt(2),  1/np.sqrt(2),  0],
                        [-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)]])
    elif idx_nv==4:
        RNV = np.array([[1/np.sqrt(6),  1/np.sqrt(6),  2/np.sqrt(6)],
                        [1/np.sqrt(2), -1/np.sqrt(2),  0],
                        [1/np.sqrt(3),  1/np.sqrt(3), -1/np.sqrt(3)]])
    else:
        raise ValueError('Invalid index of NV orientation')
    
    return RNV

def transform_vector_lab_to_NV_frame(vec_in_lab, nv_idx=1):
    """ Vector coordinates transformation from lab frame to desired NV frame.
    nv_idx can be 1, 2, 3 or 4 """
    RNV = get_rotation_matrix(nv_idx)
    vec_in_nv = np.dot(RNV, vec_in_lab)
    return vec_in_nv

def transform_vector_NV_to_lab_frame(vec_in_nv, nv_idx=1):
    """ Vector coordinates transformation from given NV frame to lab frame.
    nv_idx can be 1, 2, 3 or 4 """
    RNV = get_rotation_matrix(nv_idx)
    vec_in_lab = np.dot(RNV.T, vec_in_nv)
    return vec_in_lab

def transform_all_frames(B0, theta, phi):
    """ 
    Compute cartesian coordinates of a vecotr in all 4 NV frames, 
    based on its spherical coordinates in lab frame
    """
    Bvec = get_vector_cartesian(B0, theta, phi)
        
    # Concise version
    Bvec_list = [transform_vector_lab_to_NV_frame(Bvec, idx) 
                 for idx in range(1, 5)]
    
    return Bvec_list

# Spherical coordiantes transformation
def transform_spherical_nv_to_lab_frame(theta_nv, phi_nv, idx_nv=1):
    """ Spherical coordinates transformation from given NV frame to lab frame.
    nv_idx can be 1, 2, 3 or 4 """
    vec_in_nv = get_vector_cartesian(1, theta_nv, phi_nv)
    vec_in_lab = transform_vector_NV_to_lab_frame(vec_in_nv, idx_nv)
    _, theta_lab, phi_lab = get_vector_spherical(vec_in_lab)
    return theta_lab, phi_lab

def transform_spherical_lab_to_nv_frame(theta_lab, phi_lab, idx_nv=1):
    """ Spherical coordinates transformation from lab frame to given NV frame.
    nv_idx can be 1, 2, 3 or 4 """
    vec_in_lab = get_vector_cartesian(1, theta_lab, phi_lab)
    vec_in_nv = transform_vector_lab_to_NV_frame(vec_in_lab, idx_nv)
    _, theta_nv, phi_nv = get_vector_spherical(vec_in_nv)
    return theta_nv, phi_nv

# =============================================================================
# Single NV center Hamiltonian
# =============================================================================
# Constants 
# NV fine and hyperfine constants (in MHz)
D_0 = 2.87e3
Apar = -2.14
Aperp = -2.7
PQ = -4.95

# Magnetic coupling constants (in SI units)
muB = 9.274e-24
gNV = 2.0028
muN = 5.051e-27
gN = 0.404
h = 6.626e-34

# Gyromagnetic ratios (in MHz/G)
gammaNV = muB * gNV / h / 1e10 # NV gyromagnetic ratio 
gammaN = muN * gN / h / 1e10 # N gyromagnetic ratio (in MHz/G)

# Electric coupling constants
d_parallel = 3.5e-9 # MHz/(V/m)
d_transverse = 0.17e-6 # MHz/(V/m)

# Pauli matrices
S_x = 1 / np.sqrt(2) * np.array([[0, 1, 0],
                                 [1, 0, 1],
                                 [0, 1, 0]])
S_y = 1 / np.sqrt(2) * 1j * np.array([[0, 1, 0],
                                      [-1, 0, 1],
                                      [0, -1, 0]])
S_z = np.array([[1, 0, 0],
                [0, 0, 0], 
                [0, 0, -1]])
SI = np.eye(3)

S_zfs = np.dot(S_z, S_z) - 2/3 * SI # Matrix useful for definition of Hamiltonian

def NV_transitionsElevels(B, E):
    # Input: magnetic field and electric field, defined in NV center frame
    # Output: This function diagonalizes the Hamiltonian Hgs and return 9 eignenergies and its vector

    # Hamiltonian
    # Fine and hyperfine terms
    HZFS = D_0 * np.kron(S_zfs, SI) # Zero-field splitting
    HHFPar = Apar * np.kron(S_z, S_z) # Axial hyperfine interaction
    HHFPerp = Aperp * (np.kron(S_x, S_x) + np.kron(S_y, S_y)) # Non-axial hyperfine interaction
    HNucQ = PQ * np.kron(SI, S_zfs) # Nuclear quadrupole interaction

    # Magnetic field coupling terms
    HBEl = gammaNV * np.kron(B[0]*S_x + B[1]*S_y + B[2]*S_z, SI) # Electric Zeeman coupling
    HBNuc = gammaN * np.kron(SI, B[0]*S_x + B[1]*S_y + B[2]*S_z) # Nuclear Zeeman coupling

    # Electric field coupling terms
    H_elec = (E[2] * d_parallel * np.kron(S_zfs, SI)
              + E[0] * d_transverse * np.kron((np.dot(S_y, S_y) - np.dot(S_x, S_x)), SI)
              + E[1] * d_transverse * np.kron((np.dot(S_x, S_y) + np.dot(S_y, S_x)), SI))
    
    H_total = HZFS + HBEl + HBNuc + H_elec + HHFPar + HHFPerp + HNucQ
    E_I, vec_I = np.linalg.eigh(H_total)

    return E_I, vec_I

def NV_GS_Hamiltonian_MWprobe(Bmw):
    # Compute interaction Hamiltonian, with MW vector Bmw defined in NV center frame

    # Magnetic field coupling terms
    HintEl = gammaNV * np.kron(Bmw[0]*S_x + Bmw[1]*S_y + Bmw[2]*S_z, SI) # To electric spin
    HintNuc = gammaN * np.kron(SI, Bmw[0]*S_x + Bmw[1]*S_y + Bmw[2]*S_z) # To nuclear spin

    # Total interation Hamiltonian
    Hint = HintEl + HintNuc
    return Hint

# =============================================================================
# Computation of ODMR spectrum
# =============================================================================
def lorentzian(x, x0, fwhm):
    return 1 / (1 + (x - x0)**2 / (fwhm / 2)**2)

def ESR_singleNV(MWfreq, MWvec, Bvec, Evec, Linewidth):
    # All vectors are defined in NV frame
    nMW = len(MWfreq) # Number of frequency points
    Tstrength = np.zeros(nMW) # transition strength

    E_I, vec_I = NV_transitionsElevels(Bvec, Evec) # Eigenenergies and eigenvectors
    Hint = NV_GS_Hamiltonian_MWprobe(MWvec) # Interaction Hamiltonian

    # Calculate transition strengths
    for initS in np.arange(9): # Sweep over all initial states
        initFreq = E_I[initS] # frequency
        initVec = vec_I[:,initS]

        for finS in np.arange(initS, 9): # Sweep over all final states
            finFreq = E_I[finS] # frequency
            finVec = vec_I[:,finS] # state

            # Transition matrix element and transition amplitude
            TME = np.dot(np.dot(np.conj(finVec.transpose()),Hint), initVec)
            TA = np.abs(TME)**2
            
            # Add lorentzian lineshape
            TS = TA * lorentzian(MWfreq, abs(finFreq - initFreq), Linewidth)
            
            Tstrength += TS
                
    return Tstrength

def ESR_NVensemble(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth):
    # All vectors are defined in lab frame (spherical coordinates)
    nMW = len(MWfreq) # Number of frequency points
    Tstrength = np.zeros(nMW) # transition strength
    
    Bvector_list = transform_all_frames(B0, thetaB, phiB)
    Evector_list = transform_all_frames(E0, thetaE, phiE)
    MWvector_list = transform_all_frames(1, thetaMW, phiMW)
    
    for MWvec, Bvec, Evec in zip(MWvector_list, Bvector_list, Evector_list):
        Tstrength += ESR_singleNV(MWfreq, MWvec, Bvec, Evec, Linewidth)
        
    n_NV = len(Bvector_list) # number of NV orientations in ensemble
    return Tstrength / n_NV

def ESR_VNensemble(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth):
    return ESR_NVensemble(MWfreq, thetaMW + np.pi, phiMW, 
                          B0, thetaB + np.pi, phiB, 
                          E0, thetaE + np.pi, phiE, Linewidth)

def ESR_NV_VN_ensemble(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth):
    T_NV = ESR_NVensemble(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth)
    T_VN = ESR_VNensemble(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth)
    return (T_NV + T_VN) / 2

# =============================================================================
# Computation of Lockin ODMR spectrum
# =============================================================================
def dispersive_lineshape(x, x0, fwhm):
    xred = (x - x0) / (fwhm / 2)
    return - 2 * xred / (1 + xred**2)**2

def ESR_Lockin_singleNV(MWfreq, MWvec, Bvec, Evec, Linewidth):
    # All vectors are defined in NV frame
    nMW = len(MWfreq) # Number of frequency points
    Lockin = np.zeros(nMW) # transition strength

    E_I, vec_I = NV_transitionsElevels(Bvec, Evec) # Eigenenergies and eigenvectors
    Hint = NV_GS_Hamiltonian_MWprobe(MWvec) # Interaction Hamiltonian

    # Calculate transition strengths
    for initS in np.arange(9): # Sweep over all initial states
        initFreq = E_I[initS] # frequency
        initVec = vec_I[:,initS]

        for finS in np.arange(initS, 9): # Sweep over all final states
            finFreq = E_I[finS] # frequency
            finVec = vec_I[:,finS] # state

            # Transition matrix element and transition amplitude
            TME = np.dot(np.dot(np.conj(finVec.transpose()),Hint), initVec)
            TA = np.abs(TME)**2
            
            # Add lorentzian lineshape
            TS = TA * dispersive_lineshape(MWfreq, abs(finFreq - initFreq), Linewidth)
            
            Lockin += TS
                
    return Lockin
