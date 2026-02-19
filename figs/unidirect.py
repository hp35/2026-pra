#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generation of figures for the manuscript "Optical parametric processes in
chiral nonlinear media" by Fredrik Jonsson, Christos Flytzanis, and Govind
Agrawal. All graphs are generated as Encapsulated PostScript (.eps), Scalable
Vector Graphics (.svg) and Portable Vector Graphics (.png).

Copyright (C) 2026, Fredrik Jonsson under GPLv3.
File: copaproc/python/unidirect.py, created on Mon Oct  6 10:07:11 2025.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipj, ellipk  # Jacobi elliptic functions
from mpmath import ellippi    # Incomplete elliptic integral of third kind
from savestokes import saveStokesParameters

"""
As a global standard, use TeX-style labeling for everything graphics-related.
"""
plt.rcParams.update({
    "text.usetex" : True,
    "font.family" : "Computer Modern",
    "font.size"   : 12
})

def fieldenvelopes(
    zn = np.linspace(0.0, 1.0, num=1000),
    dkl = 0.3,
    danorm = 1.0,
    s2p = 0.1,
    s2m = 0.1,
    zetap = 6.0,
    zetam = 6.0):    # $\zeta_- = \kappa_- L \sqrt{u^+_3(0)}$
    """
    Compute the field envelopes $a^+_k(z)$ $a^-_k(z)$ for the idler (k=1),
    signal (k=2) and pump (k=3) fields. See manuscript for context and
    detailed derivation of the parameters passed as arguments to this
    function.

    Parameters
    ----------
    zn : numpy array, optional
        Normalized distance $z/L$.
        The default is np.linspace(0.0, 1.0, num=1000).
    dkl : float, optional
        Electric dipolar phase mismatch $\Delta k L$.
        The default is 0.3.
    danorm : float, optional
        Normalized gyrotropic addition $\Delta\alpha/\Delta k$.
        The default is 0.2.
    s2p : float, optional
        Initial LCP signal-to-pump ratio $s^2_+ = u^+_2(0)/u^-_3(0)$.
        The default is 0.1.
    s2m : float, optional
        Initial RCP signal-to-pump ratio $s^2_- = u^-_2(0)/u^+_3(0)$.
        The default is 0.1.
    zetap : float, optional
        Normalized LCP gain coefficient $\zeta_+ = \kappa_+ L \sqrt{u^-_3(0)}$.
        The default is 6.0.
    zetam : float, optional
        Normalized RCP gain coefficient $\zeta_- = \kappa_- L \sqrt{u^+_3(0)}$.
        The default is 6.0.

    Returns
    -------
    None.

    """

    """
    Normalized phase mismatch $\phi_{\pm}$ for LCP/RPC.
    """
    phip = dkl*(1+danorm)/(2*zetap)  # Normalized phase mismatch $\phi_+$
    phim = dkl*(1-danorm)/(2*zetam)  # Normalized phase mismatch $\phi_-$
    phi2p = np.square(phip)    # Squared normalized phase mismatch $\phi^2_+$
    phi2m = np.square(phim)    # Squared normalized phase mismatch $\phi^2_-$

    """
    Coefficients of the expressions for the evolution of the envelopes.
    """
    tmp = 1+s2p+phi2p
    u3ap_norm = (tmp - np.sqrt(tmp**2-4*phi2p))/2.0   # $u^+_{3a}/u^+_3(0)$
    u3bp_norm = 1.0                                   # $u^+_{3b}/u^+_3(0)$
    u3cp_norm = (tmp + np.sqrt(tmp**2-4*phi2p))/2.0   # $u^+_{3c}/u^+_3(0)$

    tmp = 1+s2m+phi2m
    u3am_norm = (tmp - np.sqrt(tmp**2-4*phi2m))/2.0   # $u^-_{3a}/u^-_3(0)$
    u3bm_norm = 1.0                                   # $u^-_{3b}/u^-_3(0)$
    u3cm_norm = (tmp + np.sqrt(tmp**2-4*phi2m))/2.0   # $u^-_{3c}/u^-_3(0)$

    """
    Compute the Jacobi elliptic functions describing the evolution
    of the envelopes.
    """
    xip = (u3bm_norm-u3am_norm)/(u3cm_norm-u3am_norm) # Modulus $\xi_+$
    xim = (u3bp_norm-u3ap_norm)/(u3cp_norm-u3ap_norm) # Modulus $\xi_-$
    mp = np.square(xip)       # Squared modulus $m_+ = \xi^2_+$
    mm = np.square(xim)       # Squared modulus $m_- = \xi^2_-$
    kkp = ellipk(mp)          # Elliptic integral of first kind $K(\xi_+)$
    kkm = ellipk(mm)          # Elliptic integral of first kind $K(\xi_-)$
    fgp = np.sqrt(np.sqrt(np.square(1.0+s2p+phi2p)-4.0*phi2p))
    fgm = np.sqrt(np.sqrt(np.square(1.0+s2p+phi2p)-4.0*phi2p))
    snp, cnp, dnp, php = ellipj(2*zetap*fgp*zn + kkp, mp)
    snm, cnm, dnm, phm = ellipj(2*zetam*fgm*zn + kkm, mm)

    snp2 = np.square(snp)     # $\sn^2(..., \xi_+)$
    snm2 = np.square(snm)     # $\sn^2(..., \xi_-)$

    u3p_norm = u3ap_norm + (u3bp_norm-u3ap_norm)*snp2  # $u^+_3(z)/u^+_3(0)$
    u3m_norm = u3am_norm + (u3bm_norm-u3am_norm)*snm2  # $u^-_3(z)/u^-_3(0)$

    u2p_norm = 1+s2p-u3m_norm   # $u^+_2(z)/u^-_3(0)$
    u2m_norm = 1+s2m-u3p_norm   # $u^+_2(z)/u^-_3(0)$

    u1p_norm = 1-u3m_norm   # $u^+_1(z)/u^-_3(0) = 1 - u^-_3(z)/u^-_3(0)$
    u1m_norm = 1-u3p_norm   # $u^+_1(z)/u^-_3(0) = 1 - u^+_3(z)/u^+_3(0)$

    """
    Compute the individual phase corrections $\varphi^{\pm}_k$ for the
    (k=1) idler, (k=2) signal, and (k=3) pump. We start with the rather
    extensive set of defining coefficients. Here all u3a_plus=$u^+_{3a}$,
    u3a_minus=$u^-_{3a}$ etc. are normalized against the initial pump
    envelopes $u^{\pm}_3(0)$. See the theory in the manuscript for details
    on their definition.
    """
    u3a_plus = 0.5*(1+s2p+phi2p-np.sqrt(np.square(1+s2p+phi2p)-4*phi2p))
    u3a_minus = 0.5*(1+s2m+phi2m-np.sqrt(np.square(1+s2m+phi2m)-4*phi2m))
    u3b_plus = 1
    u3b_minus = 1
    u3c_plus = 0.5*(1+s2p+phi2p+np.sqrt(np.square(1+s2p+phi2p)-4*phi2p))
    u3c_minus = 0.5*(1+s2m+phi2m+np.sqrt(np.square(1+s2m+phi2m)-4*phi2m))

    """
    Coefficients $A_{\pm}$, $B_{\pm}$, $C_{\pm}$, $D_{\pm}$ for the pump.
    """
    aa_plus = (u3b_minus-u3a_minus)/(1.0-u3a_minus)   # Constant $A_+$
    aa_minus = (u3b_plus-u3a_plus)/(1.0-u3a_plus)     # Constant $A_-$
    bb_plus = (u3b_minus-u3a_minus)/u3a_minus         # Constant $B_+$
    bb_minus = (u3b_plus-u3a_plus)/u3a_plus           # Constant $B_-$
    cc_plus = zetap*phip*(1-u3a_minus)/u3a_minus      # Constant $C_+$
    cc_minus = zetam*phim*(1-u3a_plus)/u3a_plus       # Constant $C_.$
    dd_plus = zetap*np.sqrt(u3c_minus-u3a_minus)      # Constant $D_+$
    dd_minus = zetam*np.sqrt(u3c_plus-u3a_plus)       # Constant $D_-$
    
    """
    Coefficients $A'_{\pm}$, $B'_{\pm}$, $C'_{\pm}$, $D'_{\pm}$ for the pump.
    """
    aa_prim_plus = aa_plus                                      # $A'_+$
    aa_prim_minus = aa_minus                                    # $A'_-$
    bb_prim_plus = (u3b_minus-u3a_minus)/(1+s2p-u3a_minus)      # $B'_+$
    bb_prim_minus = (u3b_plus-u3a_plus)/(1+s2m-u3a_plus)        # $B'_-$
    cc_prim_plus = zetap*phip*(1-u3a_minus)/(1+s2p-u3a_minus)   # $C'_+$
    cc_prim_minus = zetam*phim*(1-u3a_plus)/(1+s2m-u3a_plus)    # $C'_.$
    dd_prim_plus = dd_plus                                      # $D'_+$
    dd_prim_minus = dd_minus                                    # $D'_-$
    
    print("xi_+ = %1.4f, xi^2_+ = %1.4f"%(xip, xip*xip))
    print("A_+ = %1.4f, A_- = %1.4f"%(aa_plus,aa_minus))
    print("B_+ = %1.4f, B_- = %1.4f"%(bb_plus,bb_minus))
    print("C_+ = %1.4f, C_- = %1.4f"%(cc_plus,cc_minus))
    print("D_+ = %1.4f, D_- = %1.4f"%(dd_plus,dd_minus))
    print("A_+*C_+/B_+ = %1.4f, A_-*C_-/B_- = %1.4f"
          %(aa_plus*cc_plus/bb_plus,
            aa_minus*cc_minus/bb_minus))

    print("xi_- = %1.4f, xi^2_- = %1.4f"%(xim, xim*xim))
    print("A'_+ = %1.4f, A'_- = %1.4f"%(aa_prim_plus,aa_prim_minus))
    print("B'_+ = %1.4f, B'_- = %1.4f"%(bb_prim_plus,bb_prim_minus))
    print("C'_+ = %1.4f, C'_- = %1.4f"%(cc_prim_plus,cc_prim_minus))
    print("D'_+ = %1.4f, D'_- = %1.4f"%(dd_prim_plus,dd_prim_minus))
    print("A'_+*C'_+/B'_+ = %1.4f, A'_-*C'_-/B'_- = %1.4f"
          %(aa_prim_plus*cc_prim_plus/bb_prim_plus,
            aa_prim_minus*cc_prim_minus/bb_prim_minus))

    print("K_+ = %1.4f, K'_- = %1.4f"%(kkp,kkm))

    """
    Pump phase $\varphi^+_3(z)$ (LCP) and $\varphi^-_3(z)$ (RCP).
    """
    elippi_upper_plus,elippi_lower_plus = np.zeros_like(zn),np.zeros_like(zn)
    elippi_upper_minus,elippi_lower_minus = np.zeros_like(zn),np.zeros_like(zn)
    for k in range(len(zn)):
        """ Extract values for the Jacobi elliptic fcn sn(u,xi) """
        sn_upper_plus, cntmp, dntmp, phtmp = ellipj(dd_plus*zn[k]+kkp,mp)
        sn_upper_minus, cntmp, dntmp, phtmp  = ellipj(dd_minus*zn[k]+kkm,mm)
        sn_lower_plus, cntmp, dntmp, phtmp  = ellipj(kkp, mp)
        sn_lower_minus, cntmp, dntmp, phtmp  = ellipj(kkm, mm)
        pi_limit_upper_plus = np.arcsin(sn_upper_plus)
        pi_limit_upper_minus = np.arcsin(sn_upper_minus)
        pi_limit_lower_plus = np.arcsin(sn_lower_plus)
        pi_limit_lower_minus = np.arcsin(sn_lower_minus)
        pi_limit_upper_plus
        elippi_upper_plus[k] = ellippi(-bb_plus, pi_limit_upper_plus, mp)
        elippi_lower_plus[k] = ellippi(-bb_plus, pi_limit_lower_plus, mp)
        elippi_upper_minus[k] = ellippi(-bb_minus, pi_limit_upper_minus, mm)
        elippi_lower_minus[k] = ellippi(-bb_minus, pi_limit_lower_minus, mm)

    varphi3p = (aa_plus*cc_plus/bb_plus)*zn \
                  -((aa_plus+bb_plus)*cc_plus/(bb_plus*dd_plus)) \
                      *(elippi_upper_plus - elippi_lower_plus)
    varphi3m = (aa_minus*cc_minus/bb_minus)*zn \
                  -((aa_minus+bb_minus)*cc_minus/(bb_minus*dd_minus)) \
                      *(elippi_upper_minus - elippi_lower_minus)
    if np.iscomplex(varphi3p).any():
        raise ValueError('Pump phase varphi3p contains complex numbers!')
    if np.iscomplex(varphi3m).any():
        raise ValueError('Pump phase varphi3m contains complex numbers!')

    """
    Signal phase $\varphi^+_2(z)$ (LCP) and $\varphi^-_2(z)$ (RCP).
    """
    elippi_upper_plus,elippi_lower_plus = np.zeros_like(zn),np.zeros_like(zn)
    elippi_upper_minus,elippi_lower_minus = np.zeros_like(zn),np.zeros_like(zn)
    for k in range(len(zn)):
        """ Extract values for the Jacobi elliptic fcn sn(u,xi) """
        sn_upper_plus, cntmp, dntmp, phtmp = ellipj(dd_prim_plus*zn[k]+kkp,mp)
        sn_upper_minus, cntmp, dntmp, phtmp = ellipj(dd_prim_minus*zn[k]+kkm,mm)
        sn_lower_plus, cntmp, dntmp, phtmp = ellipj(kkp, mp)
        sn_lower_minus, cntmp, dntmp, phtmp = ellipj(kkm, mm)
        pi_limit_upper_plus = np.arcsin(sn_upper_plus)
        pi_limit_upper_minus = np.arcsin(sn_upper_minus)
        pi_limit_lower_plus = np.arcsin(sn_lower_plus)
        pi_limit_lower_minus = np.arcsin(sn_lower_minus)
        pi_limit_upper_plus
        elippi_upper_plus[k] = ellippi(bb_prim_plus, pi_limit_upper_plus, mp)
        elippi_lower_plus[k] = ellippi(bb_prim_plus, pi_limit_lower_plus, mp)
        elippi_upper_minus[k] = ellippi(bb_prim_minus, pi_limit_upper_minus, mm)
        elippi_lower_minus[k] = ellippi(bb_prim_minus, pi_limit_lower_minus, mm)

    varphi2p = (aa_prim_plus*cc_prim_plus/bb_prim_plus)*zn \
                  -((aa_prim_plus-bb_prim_plus)*cc_prim_plus \
                    /(bb_prim_plus*dd_prim_plus)) \
                      *(elippi_upper_plus - elippi_lower_plus)
    varphi2m = (aa_prim_minus*cc_prim_minus/bb_prim_minus)*zn \
                  -((aa_prim_minus-bb_prim_minus)*cc_prim_minus \
                    /(bb_prim_minus*dd_prim_minus)) \
                      *(elippi_upper_minus - elippi_lower_minus)
    if np.iscomplex(varphi2p).any():
        raise ValueError('Signal phase varphi2p contains complex numbers!')
    if np.iscomplex(varphi2m).any():
        raise ValueError('Signal phase varphi2m contains complex numbers!')

    """
    Compute the complex-valued field envelopes a^{\pm}_k for the idler (k=1),
    signal (k=2) and pump (k=3).
    """
    a3_plus = np.multiply(np.sqrt(u3p_norm),np.exp(1j*varphi3p))  # Pump, LCP
    a3_minus = np.multiply(np.sqrt(u3m_norm),np.exp(1j*varphi3m)) # Pump, RCP
    a2_plus = np.multiply(np.sqrt(u2p_norm),np.exp(1j*varphi2p))  # Signal, LCP
    a2_minus = np.multiply(np.sqrt(u2m_norm),np.exp(1j*varphi2m)) # Signal, RCP

    return a3_plus, a3_minus, u3p_norm, u3m_norm, varphi3p, varphi3m, \
                a2_plus, a2_minus, u2p_norm, u2m_norm, varphi2p, varphi2m

def stokeparams(a_plus, a_minus):
    """
    Compute the Stokes parameters corresponding to the supplied complex-valued
    field envelopes, as evolving along the normalized z-axis (z/L).
    """
    s0 = np.square(np.abs(a_plus))+np.square(np.abs(a_minus))
    s3_norm = np.square(np.abs(a_plus))-np.square(np.abs(a_minus))
    s3_norm = np.divide(s3_norm, s0)
    s1_norm = 2.0*np.real(np.multiply(np.conj(a_plus),a_minus))
    s1_norm = np.divide(s1_norm, s0)
    s2_norm = 2.0*np.imag(np.multiply(np.conj(a_plus),a_minus))
    s2_norm = np.divide(s2_norm, s0)
    return s0, s1_norm, s2_norm, s3_norm

def plotfieldenvelopes(zn, s0, s1_norm, s2_norm, s3_norm,
                       varphi_plus, varphi_minus,
                       varphi_plus_text, varphi_minus_text, titletext):
    """
    Plot the extracted Stokes parameters for the pump.
    """
    fig, ax = plt.subplots(4,1,figsize=(10.0,7.0))
    major_x_ticks = np.linspace(0.0, 1.0, 6)

    ax[0].plot(zn, s0/s0[0], 'k', label="$S_0(z)$")
    ax[0].autoscale(enable=True, axis='x', tight=True)
    ax[0].set_ylim(bottom=0.0, top=np.max(s0/s0[0])+0.1)
    ax[0].set_ylabel('$S_0(z)/S_0(0)$')
    ax[0].set_xticks(major_x_ticks)
#    ax[0].set_yticks(np.linspace(0.0, 1.0, 5))
    ax[0].grid(which='both')
    ax[0].set_title(titletext)

    ax[1].plot(zn, s3_norm, 'k', label="$S_3(z)/S_0(z)$")
    ax[1].autoscale(enable=True, axis='x', tight=True)
    ax[1].set_ylim(bottom=-1.0, top=1.0)
    ax[1].set_ylabel('$S_3(z)/S_0(z)$')
    ax[1].set_xticks(major_x_ticks)
    ax[1].set_yticks(np.linspace(-1.0, 1.0, 5))
    ax[1].grid(which='both')

    ax[2].plot(zn, varphi_plus, 'r', label=varphi_plus_text)
    ax[2].plot(zn, varphi_minus, 'b', label=varphi_minus_text)
    ax[2].autoscale(enable=True, axis='x', tight=True)
    ax[2].legend(loc="upper right")
    ax[2].set_xticks(major_x_ticks)
#    ax[2].set_yticks(np.linspace(0.0, 1.0, 5))
    ax[2].grid()
    ax[2].set_ylabel('$\\varphi^{\pm}(z)\ ({\\rm phase})$')

    ax[3].plot(zn, s1_norm, 'r', label="$S_1(z)/S_0(z)$")
    ax[3].plot(zn, s2_norm, 'g', label="$S_2(z)/S_0(z)$")
    ax[3].plot(zn, s3_norm, 'b', label="$S_3(z)/S_0(z)$")
    ax[3].plot(zn, np.square(s1_norm)+np.square(s2_norm)
           +np.square(s3_norm), 'k--', label="$\sum S^2_k(z)/S^2_0(z)$")
    ax[3].autoscale(enable=True, axis='x', tight=True)
    ax[3].legend(loc="upper right")
    ax[3].set_xticks(major_x_ticks)
    ax[3].set_yticks(np.linspace(-1.0, 1.0, 5))
    ax[3].grid()
    ax[3].set_ylabel('$S_k(z)/S_0(z)$')
    ax[3].set_xlabel('$z/L$')
    return fig

def savegraph(fig, basename="out"):
    kwargs={'bbox_inches':'tight', 'pad_inches':0.0}
    for fmt in ['eps','svg','png']:
        fig.savefig(basename+'.'+fmt, format=fmt, **kwargs)
    return

def graph_06():
    """
    Everything is in this set of graphs mapped over the normalized distance
    zn = z/L. L is in turn included in parameters such as the normalized gain,
    phase mismatch etc., and never explicitly stated as such.
    """
    zn = np.linspace(0.0, 1.0, num=500)
    dkl = 2.0       # Electric dipolar phase mismatch $\Delta k L$
    danorm = 2.0    # Normalized gyrotropic addition $\Delta\alpha/\Delta k$

    """
    Initial signal-to-pump ratio for LCP/RPC.
    """
    s2p = 0.1       # $s^2_+ = u^+_2(0)/u^-_3(0)$
    s2m = 0.1       # $s^2_- = u^-_2(0)/u^+_3(0)$

    """
    Normalized gain coefficients $\zeta_+$ and $\zeta_-$ for LCP/RPC.
    """
    zetap = 4.0       # $\zeta_+ = \kappa_+ L \sqrt{u^-_3(0)}$
    zetam = 4.0       # $\zeta_- = \kappa_- L \sqrt{u^+_3(0)}$

    """
    Compute and plot the field envelopes and Stokes parameters corresponding
    to the pump field, as evolving along the normalized z-axis (z/L).
    """
    a3_plus, a3_minus, u3p_norm, u3m_norm, varphi3p, varphi3m, \
    a2_plus, a2_minus, u2p_norm, u2m_norm, varphi2p, varphi2m  \
        = fieldenvelopes(zn, dkl, danorm, s2p, s2m, zetap, zetam)

    s0_pump, s1_pump_norm, s2_pump_norm, s3_pump_norm \
        = stokeparams(a3_plus, a3_minus)
    fig = plotfieldenvelopes(zn,
                s0_pump, s1_pump_norm, s2_pump_norm, s3_pump_norm,
                varphi3p, varphi3m,
                "$\\varphi^+_3(z)$", "$\\varphi^-_3(z)$", "Pump")
    savegraph(fig, basename="graph-06-pump")
    saveStokesParameters(zn, s0_pump, s1_pump_norm, s2_pump_norm, s3_pump_norm,
                "graph-06-stokes-signal", fmt="%f", delimiter=" ", header="p", footer="q",
                saveformat="poincare", tickspacing=0.05,
                ticklabel=False, ticklabelspacing=0.2)

    s0_signal, s1_signal_norm, s2_signal_norm, s3_signal_norm \
        = stokeparams(a2_plus, a2_minus)
    fig = plotfieldenvelopes(zn,
                s0_signal, s1_signal_norm, s2_signal_norm, s3_signal_norm,
                varphi2p, varphi2m,
                "$\\varphi^+_2(z)$", "$\\varphi^-_2(z)$", "Signal")
    savegraph(fig, basename="graph-06-signal")
    saveStokesParameters(zn, s0_signal, s1_signal_norm, s2_signal_norm, s3_signal_norm,
                "graph-06-stokes-pump", fmt="%f", delimiter=" ", header="p", footer="q",
                saveformat="poincare", tickspacing=0.05,
                ticklabel=False, ticklabelspacing=0.2)

    return

def main() -> None:
    graph_06()
    return

if __name__ == "__main__":
    main()#!/usr/bin/env python3
