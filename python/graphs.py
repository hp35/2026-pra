# -*- coding: utf-8 -*-
"""
Generation of figures for the manuscript "Optical parametric processes in
chiral nonlinear media" by Fredrik Jonsson, Christos Flytzanis, and Govind
Agrawal. All graphs are generated as Encapsulated PostScript (.eps), Scalable
Vector Graphics (.svg) and Portable Vector Graphics (.png).

Copyright (C) 2025, Fredrik Jonsson under GPLv3.
File: copaproc/python/graphs.py, created on Sat May 3 20:01:54 2025.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipj, ellipk
from scipy.optimize import fsolve
import math
from matplotlib.ticker import AutoLocator, AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from scipy.interpolate import griddata

"""
As a global standard, use TeX-style labeling for everything graphics-related.
"""
plt.rcParams.update({
    "text.usetex" : True,
    "font.family" : "Computer Modern",
    "font.size"   : 12
})

colors=['xkcd:red','xkcd:green','xkcd:azure','xkcd:tan','xkcd:teal']
#dashdotdotted = (0, (5, 1, 1, 1, 1, 1))  # Custom '—··—··—··—'
linestyles=['dashdot','dotted','dashed','solid']
gridcolor=[0.85,0.85,0.85]
boxcolor=[0.50,0.50,0.50]

def addlegend(ax, ccolors, clinestyles, clabels):
    linespacing = 0.055     # Spacing between legends in the legend box
    linelength = 0.08       # Length of displayed lines in the legend box
    xa, ya = 0.70, 0.78     # Lower-left corner of the legend box
    boxwidth = 0.29         # Width of the rectangular legend box
    boxheight = 0.22        # Height of the rectangular legend box
    linetextspacing = 0.02  # Horizontal spacing between line and text
    alpha = 0.90            # Transparency of the legend text box
    zorder = 5              # Z-order of legend text box
    fontsize = 10           # Font size of the text in the legend box
    xb = xa + linelength
    rect=patches.Rectangle((xa-0.02,ya-0.02), boxwidth, boxheight, angle=0.0, 
               rotation_point='xy', facecolor='white', edgecolor=boxcolor, 
               capstyle='round', joinstyle='round',
               linestyle='-', linewidth=1, fill=True, alpha=alpha,
               transform=ax.transAxes, zorder=zorder)
    ax.add_patch(rect)
    for k in range(len(ccolors)):
        y = ya + k*linespacing
        ax.plot([xa,xb],[y,y],color=ccolors[k],linestyle=clinestyles[k],
                transform=ax.transAxes, zorder=zorder, 
                solid_capstyle='round', dash_capstyle='round')
        ax.text(xb+linetextspacing, y, clabels[k], transform=ax.transAxes,
                fontsize=fontsize,
                horizontalalignment='left', verticalalignment='center', 
                zorder=zorder)
    return

def implicittest(plotf=True):

    def f(zeta,s):
        return np.square(zeta)*np.exp(-np.square(zeta-1)) - s

    zetamax = (1.0+np.sqrt(5.0))/2.0
    smax = f(zetamax,0.0)
    print("zetamax=%1.4f, smax=%1.4f"%(zetamax,smax))

    if plotf:
        """
        The regular function plot of s(zeta).
        """
        zeta = np.linspace(0.0,5.0,1000)
        s = np.square(zeta)*np.exp(-np.square(zeta-1))
        fig, ax = plt.subplots(figsize=(5.2,4.0))
        ax.plot(zeta, s, color='#1f77b4ff', linewidth=1.5)
        ax.plot(zetamax, smax, '+', color='red', linewidth=1.5)
        ax.set_xlabel('$\\zeta$')
        ax.set_ylabel('$s$')
        ax.set_title('Function $f(s)$ used for optimization')
        ax.grid(visible=True, which='major', axis='both', color=gridcolor)
        ax.tick_params(which="both", top=True, right=True, labeltop=False, 
                       bottom=True, labelbottom=True, direction="in")

        """
        The function surface f(\zeta,s).
        """
        fig, ax = plt.subplots(figsize=(5.4,5.4),
                               subplot_kw={"projection":"3d"})
        zeta = np.linspace(0.0,5.0,1000)
        s = np.linspace(0.0,5.0,1000)
        zetag, sg = np.meshgrid(zeta,s)
        z = np.square(zetag)*np.exp(-np.square(zetag-1))-sg
        ax.plot_surface(zetag, sg, z)
        ax.contour(zetag, sg, z, 10, lw=3, cmap="autumn_r", 
                   linestyles="solid", offset=-1)
        ax.set_zlabel("$z=f(\\zeta,s)$")
        ax.set_xlabel("$\\zeta$")
        ax.set_ylabel("$s$")

    return

def sfunc(zetanorm, phi, rr):

    def f(zetanorm, s, phi, rr):
        zetanorm2 = np.square(zetanorm)
        phi2 = np.square(phi)
        s2 = np.square(s)
        f2 = np.sqrt(np.square(1.0+s2+phi2)-4.0*phi2)
        f = np.sqrt(f2)
        xi = np.divide(1.0-s2-phi2+f2,f2)/2.0
        m = np.square(xi)
        kk = ellipk(m)
        snk, cnk, dnk, phk = ellipj(f*np.sqrt(zetanorm2)+kk, m)
        z = (1.0-s2-phi2+f2)/2.0
        z *= np.square(cnk)
        z -= ((1.0-rr)/rr)*s2
        return z

    func = lambda s : f(zetanorm, s, phi, rr)

    s_ini = 2.0
    ssol = fsolve(func, s_ini)
    return ssol[0]

def zetathresh(phi, rr):
    """
    Compute the threshold value $\zeta^{({\rm th})}$ of the normalized
    parameter $\zeta$.
    """
    phi2 = np.square(phi)
    zetath = np.divide(np.arccosh(0j+np.sqrt(0j+(1-(1-rr)*phi2)/rr)),
                       np.sqrt(1-phi2))
    zetath = np.real(zetath)
    return zetath

def xifunc(s, phi):
    """
    Compute the modulus of the involved elliptic functions and elliptic
    integrals. Please be aware that the SciPy functions ellipj and ellipk
    use the square $m=\\xi^2$ as their parameter. (Easily confused!)
    """
    s2, phi2 = np.square(s), np.square(phi)
    f2 = np.sqrt(0j+np.square(1.0+s2+phi2)-4.0*phi2)
    u3a = (1.0+s2+phi2-f2)/2.0
    u3b = 1.0
    u3c = (1.0+s2+phi2+f2)/2.0
    xi = np.sqrt(0j+np.divide(u3b-u3a, u3c-u3a))
    return np.real(xi)

def smax(phi, rr):
    phi2 = np.square(phi)
    # cc = 1.0/(2*(1.0-rr))
    # return np.sqrt(cc*rr*((1.0+phi2)*rr+(1.0-phi2)*(2.0-rr)))
    cc = rr/(1.0-rr)
    return np.sqrt(cc*(1.0-(1.0-rr)*phi2))

def zetamax(s, phi):
    s2, phi2 = np.square(s), np.square(phi)
    m = np.square(xifunc(s, phi))
    kk = ellipk(m)
    f2 = np.sqrt(0j+np.square(1.0+s2+phi2)-4.0*phi2)
    f = np.sqrt(f2)
    return np.real(np.divide(kk,f))

def szetamax(phi, rr, plotf=True, plotcntr=True):
    smaxx = smax(phi, rr)
    zetamaxx = zetamax(smaxx, phi)
    return zetamaxx, smaxx

def savegraph(fig, basename="out"):
    kwargs={'bbox_inches':'tight', 'pad_inches':0.0}
    for fmt in ['eps','svg','png']:
        fig.savefig(basename+'.'+fmt, format=fmt, **kwargs)
    return

"""
graph_00 - Generate graph of signal-to-pump ratio $s_{\pm}$
           vs normalized phase mismatch $\phi_{\pm}$ for a
           set of reflectances $R_{\pm}$.
"""
def graph_00(rrvals, plotsurface=False, basename="whatever"):
    zeta = 1.0
    phimax = 3.0
    smax = np.sqrt(6.0)
    phi = np.linspace(-phimax, phimax, 1024)
    s = np.linspace(-smax/40, smax, 1024)

    """
    Compute the functional surface z=f(x,y) of the implicit relation
    f(x,y) which we wish to map for solutions fulfilling f(x,y)=0.
    """
    s2 = np.square(s)
    phig, sg = np.meshgrid(phi,s)
    phig2 = np.square(phig)
    sg2 = np.square(sg)
    fg2 = np.sqrt(np.square(1.0+sg2+phig2)-4.0*phig2)
    fg = np.sqrt(fg2)
    xi = np.divide(1.0-sg2-phig2+fg2,fg2)/2.0
    m = np.square(xi)
    kk = ellipk(m)
    sn, cn, dn, ph = ellipj(fg*zeta+kk, m)

    fig, ax = plt.subplots(figsize=(5.2,4.0))
    ccolors, clinestyles, clabels = [], [], []
    for k, rr in enumerate(rrvals):
        color,linestyle,labeltext = 'k',linestyles[k],'$R_{\\pm}=%1.2f$'%(rr)
        z = (1-sg2-phig2+fg2)/2.0
        z *= np.square(cn)
        z -= ((1.0-rr)/rr)*sg2
        ax.contour(phig, sg2, z, [0], zdir='z', colors=color,
                        linestyles=linestyle,
                        solid_capstyle='round', dash_capstyle='round')
        ccolors.append(color)
        clinestyles.append(linestyle)
        clabels.append(labeltext)

    addlegend(ax, ccolors, clinestyles, clabels)
    ax.grid(visible=True, which='major', axis='both', color=gridcolor)
    ax.tick_params(which="both", top=True, right=True, labeltop=False,
                   bottom=True, labelbottom=True, direction="in")
    ax.set_xlim([min(phi), max(phi)])
    ax.set_ylim([0, max(s2)])
    ax.set_xlabel("$\\phi_{\\pm}$")
    ax.set_ylabel("$s^2_{\\pm}$")
    savegraph(fig, basename=basename)

    if plotsurface:
        fig, ax = plt.subplots(figsize=(5.4,5.4),
                               subplot_kw={"projection":"3d"})
        for rr in rrvals:
            z = (1-sg2-phig2+fg2)/2.0
            z *= np.square(cn)
            z -= ((1.0-rr)/rr)*sg2
            ax.plot_surface(phig, sg2, z)
        ax.set_zlabel("$z$")
        ax.set_xlabel("$\\phi_{\\pm}$")
        ax.set_ylabel("$s^2_{\\pm}$")
        savegraph(fig, basename=basename+"-surf")
    
    return

def sum_nan_arrays(a,b):
    ma = np.isnan(a)
    mb = np.isnan(b)
    return np.where(ma&mb, np.nan, np.where(ma,0,a) + np.where(mb,0,b))

"""
graph_01 - Generate graph of signal-to-pump ratio $s_{\pm}$
           vs normalized phase mismatch $\phi_{\pm}$ for a
           set of reflectances $R_{\pm}$.
"""
def graph_01(rrvals, plotsurface=False, useLinearInterpolation=False, 
             basename="whatever"):
    zeta = 1.0
    phimax = 3.0
    smax = np.sqrt(6.0)
    phimean = 2.0
    dphi = np.linspace(-phimax, phimax, 1024)
    s = np.linspace(-smax/40, smax, 1024)

    """
    Compute the functional surface z=f(x,y) of the implicit relation
    f(x,y) which we wish to map for solutions fulfilling f(x,y)=0.
    """
    s2 = np.square(s)
#    deltaphi=1.0
    dphig, sg = np.meshgrid(dphi,s)
    phig2plus = np.square(phimean+dphig)
    phig2minus = np.square(phimean-dphig)
    sg2 = np.square(sg)
    fg2plus = np.sqrt(np.square(1.0+sg2+phig2plus)-4.0*phig2plus)
    fg2minus = np.sqrt(np.square(1.0+sg2+phig2minus)-4.0*phig2minus)
    fgplus = np.sqrt(fg2plus)
    fgminus = np.sqrt(fg2minus)
    xiplus = np.divide(1.0-sg2-phig2plus+fg2plus,fg2plus)/2.0
    ximinus = np.divide(1.0-sg2-phig2minus+fg2minus,fg2minus)/2.0
    mplus = np.square(xiplus)
    mminus = np.square(ximinus)
    kkplus = ellipk(mplus)
    kkminus = ellipk(mminus)
    snplus, cnplus, dnplus, phplus = ellipj(fgplus*zeta+kkplus, mplus)
    snminus, cnminus, dnminus, phminus = ellipj(fgminus*zeta+kkminus, mminus)

    fig, ax = plt.subplots(figsize=(5.2,4.0))
    figg, axg = plt.subplots(nrows=2, ncols=1, figsize=(5.2,4.0))
    figgg, axgg = plt.subplots(figsize=(5.2,4.0))
    ccolors, clinestyles, clabels = [], [], []
    for k, rr in enumerate(rrvals):
        color,linestyle,labeltext = 'k',linestyles[k],'$R_{\\pm}=%1.2f$'%(rr)
        zplus = (1-sg2-phig2plus+fg2plus)/2.0
        zplus *= np.square(cnplus)
        zplus -= ((1.0-rr)/rr)*sg2
        zminus = (1-sg2-phig2minus+fg2minus)/2.0
        zminus *= np.square(cnminus)
        zminus -= ((1.0-rr)/rr)*sg2
        csplus = ax.contour(dphig, sg2, zplus, [0], zdir='z', colors=color,
                        linestyles=linestyle,
                        solid_capstyle='round', dash_capstyle='round')
        csminus = ax.contour(dphig, sg2, zminus, [0], zdir='z', colors=color,
                        linestyles=linestyle,
                        solid_capstyle='round', dash_capstyle='round')
        ccolors.append(color)
        clinestyles.append(linestyle)
        clabels.append(labeltext)

        """
        Gather all contour vertices. After this, xc and yc contain the
        contour coordinates as scattered points.
        """
        nn = 10000
        x = np.linspace(np.min(dphi), np.max(dphi), nn)
        vertsplus = np.vstack([p.vertices for p in csplus.get_paths()])
        xcplus = vertsplus[:,0]     #  == dphig
        ycplus = vertsplus[:,1]     #  == sg2
        vertsminus = np.vstack([p.vertices for p in csminus.get_paths()])
        xcminus = vertsminus[:,0]   #  == dphig
        ycminus = vertsminus[:,1]   #  == sg2

        """
        Interpolate onto the grid with griddata.
        """
        pointsplus = xcplus.reshape(-1,1)     # independent variable
        valuesplus = ycplus                   # dependent variable
        pointsminus = xcminus.reshape(-1,1)   # independent variable
        valuesminus = ycminus                 # dependent variable
        if useLinearInterpolation:
            yplus = griddata(pointsplus, valuesplus, x.reshape(-1,1),
                             method='linear')
            yminus = griddata(pointsminus, valuesminus, x.reshape(-1,1),
                             method='linear')
        else:
            yplus = griddata(pointsplus, valuesplus, x.reshape(-1,1),
                         method='cubic')
            yminus = griddata(pointsminus, valuesminus, x.reshape(-1,1),
                         method='cubic')
        s0signal = sum_nan_arrays(yplus, yminus)
        s3signal = sum_nan_arrays(yplus, -yminus)
        axg[0].plot(x,s0signal, color=color,
                linestyle=linestyle,
                solid_capstyle='round', dash_capstyle='round');
        axg[1].plot(x,s3signal, color=color,
                linestyle=linestyle,
                solid_capstyle='round', dash_capstyle='round');
        axgg.plot(x,yplus,'r-',x,yminus,'b-');

    addlegend(ax, ccolors, clinestyles, clabels)
    ax.grid(visible=True, which='major', axis='both', color=gridcolor)
    ax.tick_params(which="both", top=True, right=True, labeltop=False,
                   bottom=True, labelbottom=True, direction="in")
    ax.set_xlim([min(dphi), max(dphi)])
    ax.set_ylim([0, max(s2)])
    ax.set_xlabel("$\\phi_{\\pm}$")
    ax.set_ylabel("$s^2_{\\pm}$")

    addlegend(axg[0], ccolors, clinestyles, clabels)
    axg[0].grid(visible=True, which='major', axis='both', color=gridcolor)
    axg[1].tick_params(which="both", top=True, right=True, labeltop=False,
                   bottom=True, labelbottom=True, direction="in")
    axg[0].set_xlim([min(dphi), max(dphi)])
    axg[0].set_ylim([0, max(s2)])
    axg[0].set_xlabel("$\\phi_{\\pm}$")
    axg[0].set_ylabel("$s^2_{+}+s^2_{-}$")

    savegraph(figg, basename=basename)

    if plotsurface:
        fig, ax = plt.subplots(figsize=(5.4,5.4),
                               subplot_kw={"projection":"3d"})
        for rr in rrvals:
            zplus = (1-sg2-phig2plus+fg2plus)/2.0
            zplus *= np.square(cnplus)
            zplus -= ((1.0-rr)/rr)*sg2
            zminus = (1-sg2-phig2minus+fg2minus)/2.0
            zminus *= np.square(cnminus)
            zminus -= ((1.0-rr)/rr)*sg2
            ax.plot_surface(dphig, sg2, zplus)
            ax.plot_surface(dphig, sg2, zminus)
        ax.set_zlabel("$z$")
        ax.set_xlabel("$\\phi_{\\pm}$")
        ax.set_ylabel("$s^2_{\\pm}$")
        savegraph(fig, basename=basename+"-surf")
    
    return

""""
graph_02 - Generate graph of pump threshold $\zeta_{{\rm th}\pm}$
           vs normalized phase mismatch $\phi_{\pm}$ for a set of
           reflectances $R_{\pm}$.
"""
def graph_02(rrvals, basename="whatever"):

    def carccosh(z):
        """
        https://mathworld.wolfram.com/InverseHyperbolicCosine.html
        """
        return np.log(z+np.sqrt(z+1)*np.sqrt(z-1))

    def zetathresh(rr, phi):
        phi2 = np.square(phi)
        zetath = np.arccosh(np.sqrt((1.0-(1.0-rr)*phi2)/rr+0j)+0j)
        zetath = np.divide(zetath, np.sqrt(1.0-phi2+0j))
        zetath = np.absolute(zetath)
        return zetath

    fig, ax = plt.subplots(figsize=(5.2,4.0))
    ccolors, clinestyles, clabels = [], [], []
    for k, rr in enumerate(rrvals):
        color,linestyle,labeltext = 'k',linestyles[k],'$R_{\\pm}=%1.2f$'%(rr)
        phimax = 1.0/np.sqrt(1.0-rr)-1.0e-6
        phi = np.linspace(-phimax, phimax, 1024)
        zetath = zetathresh(rr, phi)
        ax.plot(phi, zetath, color=color, linestyle=linestyle,
                solid_capstyle='round', dash_capstyle='round')
        ccolors.append(color)
        clinestyles.append(linestyle)
        clabels.append(labeltext)

    rrlimmax = 1-1/(5.0*5.0)
    rrlim = np.linspace(np.min(rrvals), rrlimmax, 1024)
    philim = np.divide(1.0,np.sqrt(1.0-rrlim))
    zetathlim = zetathresh(rrlim, philim)
    ax.plot(philim, zetathlim, color='lightgrey', linestyle='-')
    ax.plot(-philim, zetathlim, color='lightgrey', linestyle='-')

    addlegend(ax, ccolors, clinestyles, clabels)
    ax.grid(visible=True, which='major', axis='both', color=gridcolor)
    ax.tick_params(which="both", top=True, right=True, labeltop=False, 
                   bottom=True, labelbottom=True, direction="in")
    ax.set_xlim([-5.0, 5.0])
    ax.set_ylim([0.2, 0.7])
    ax.set_xlabel("$\\phi_{\\pm}$")
    ax.set_ylabel("$\\zeta^{({\\rm th})}_{\\pm}$")
    savegraph(fig, basename=basename)
    return

""""
graph_03 - Generate graph of signal-to-pump ratio $s_{\pm}$ vs normalized
           spatial coordinate $\zeta_{\pm}$ for a set of reflectances
           $R_{\pm}$. The set of graphs in this figure are generated
           under the assumption of perfect phase matching ($\phi_{\pm}=0$).
"""
def graph_03(rrvals, plotsurface=False, normalizezeta=False, logzeta=True,
             basename="whatever"):
    if normalizezeta:
        zetanormmax = 25.0
    else:
        zetanormmax = 20.0
    smax = 4.5
    phi = 0.0
    phi2 = np.square(phi)
    if logzeta:
        zetanorm = np.logspace(-np.log10(zetanormmax),
                               np.log10(zetanormmax), num=1024, base=10.0)
        # zetanorm = np.logspace(-np.log10(zetanormmax/10),
        #                        np.log10(zetanormmax), 1024)
    else:
        zetanorm = np.linspace(-zetanormmax/10, zetanormmax, 1024)
    s = np.linspace(-smax/10, smax, 1024)

    """
    Compute the functional surface z=f(x,y) of the implicit relation
    f(x,y) which we wish to map for solutions folfilling f(x,y)=0.
    """
    zetanorm2 = np.square(zetanorm)
    s2 = np.square(s)
    zetanormg2, sg2 = np.meshgrid(zetanorm2,s2)
    fg2 = np.sqrt(np.square(1.0+sg2+phi2)-4.0*phi2)
    fg = np.sqrt(fg2)
    xi = np.divide(1.0-sg2-phi2+fg2,fg2)/2.0
    sg = np.sqrt(sg2)
    xi = xifunc(sg, phi)
    print(xi)

    fig, ax = plt.subplots(figsize=(5.2,4.0))
    ccolors, clinestyles, clabels = [], [], []
    for k, rr in enumerate(rrvals):
        zetath = zetathresh(phi,rr)
        m = np.square(xi)
        kk = ellipk(m)
        if normalizezeta:
            snk, cnk, dnk, phk = ellipj(fg*zetath*np.sqrt(zetanormg2)+kk, m)
        else:
            snk, cnk, dnk, phk = ellipj(fg*np.sqrt(zetanormg2)+kk, m)
        color,linestyle,labeltext = 'k',linestyles[k],'$R_{\\pm}=%1.2f$'%(rr)
        z = (1-sg2-phi2+fg2)/2.0
        z *= np.square(cnk)
        # Alternative take, using sn²(X,ξ) rather than cn²(X+K,ξ), giving
        # the same numerical result:
        #   xi2 = xi*xi
        #   snp, cnp, dnp, php = ellipj(fg*np.sqrt(zetanormg2), xi)
        #   snp2 = np.square(snp)
        #   z *= (1-xi2)*np.divide(snp2,1-xi2*snp2)
        z -= ((1.0-rr)/rr)*sg2
        ax.contour(zetanormg2, sg2, z, [0], zdir='z', colors=color,
                        linestyles=linestyle, 
                        solid_capstyle='round', dash_capstyle='round')
        ccolors.append(color)
        clinestyles.append(linestyle)
        clabels.append(labeltext)
        zetamax, smax = szetamax(phi,rr)
        if normalizezeta:
            ax.plot(np.square(zetamax/zetath), np.square(smax), '+', color='red')
        else:
            ax.plot(np.square(zetamax), np.square(smax), '+', color='red')

    addlegend(ax, ccolors, clinestyles, clabels)
    ax.grid(visible=True, which='major', axis='both', color=gridcolor)
    ax.tick_params(which="both", top=True, right=True, labeltop=False, 
                   bottom=True, labelbottom=True, direction="in")
    
    ax.set_xscale('log')
    
    if normalizezeta:
        ax.set_xlim([1, max(zetanorm)])
    else:
        ax.set_xlim([0.04, max(zetanorm)])
    ax.set_ylim([0, max(s2)])
    ax.set_xlabel("$(\\zeta_{\\pm}/\\zeta^{({\\rm th})}_{\\pm})^2$")
    ax.set_ylabel("$s^2_{\\pm}$")
    savegraph(fig, basename=basename)

    if plotsurface:
        fig, ax = plt.subplots(figsize=(5.4,5.4),
                               subplot_kw={"projection":"3d"})
        for rr in rrvals:
            z = (1-sg2-phi2+fg2)/2.0
            z *= np.square(cnk)
            z -= ((1.0-rr)/rr)*sg2
            ax.plot_surface(zetanormg2, sg2, z)
        ax.set_zlabel("$z$")
        ax.set_xlabel("$(\\zeta_{\\pm}/\\zeta^{({\\rm th})}_{\\pm})^2$")
        ax.set_ylabel("$s^2_{\\pm}$")
        savegraph(fig, basename=basename+"-surf")
    
    return


""""
graph_04 - Generate graph of constant levels of intensity as function of
           non-local (chiral) phase-mismatch along the x-axis and the
           electric dipolar (local) phase-mismatch along the y-axis, at
           a fixed normalized spatial coordinate $\zeta_{\pm}$ and for a
           set of reflectances $R_{\pm}$.
"""
def graph_04(rrvals, plotsurface=False, basename="whatever"):
    zetanorm = 1.2
    phimax = 2.0
    n = 100
    phip, phim = np.linspace(-phimax,phimax,n), np.linspace(-phimax,phimax,n)
    phipg, phimg = np.meshgrid(phip, phim)
    spg, smg = np.zeros_like(phipg), np.zeros_like(phimg)

    for k, rr in enumerate(rrvals):
        """
        Generate the datasets over grid, to be used for surface and contour
        visualization.
        """
        for j1 in range(n):
            for j2 in range(n):
                spg[j1,j2] = sfunc(zetanorm, phipg[j1,j2], rr)
                smg[j1,j2] = sfunc(zetanorm, phimg[j1,j2], rr)

        s0g, s3g = np.zeros_like(spg), np.zeros_like(spg)
        s0g = np.add(np.square(spg),np.square(smg))
        s3g = np.divide(np.subtract(np.square(spg),np.square(smg)),s0g)
        philocg = phipg+phimg
        phinonlocg = phipg-phimg

        fig, ax = plt.subplots(figsize=(5.2,4.0))
        ax.contour(phinonlocg, philocg, s3g, zdir='z', colors='k',
                        linestyles='solid', 
                        solid_capstyle='round', dash_capstyle='round')

        ax.set_aspect("equal")
        ax.grid(visible=True, which='major', axis='both', color=gridcolor)
        ax.tick_params(which="both", top=True, right=True, labeltop=False, 
                       bottom=True, labelbottom=True, direction="in")
    #    ax.set_xlim([0, max(zetanorm)])
    #    ax.set_ylim([0, max(s2)])
        ax.set_xlabel("$\\phi_{+}+\\phi_{-}$")
        ax.set_ylabel("$\\phi_{+}-\\phi_{-}$")
        savegraph(fig, basename=basename)

        if plotsurface:
            fig, ax = plt.subplots(figsize=(5.4,5.4),
                                   subplot_kw={"projection":"3d"})
            ax.plot_surface(phinonlocg, philocg, s3g)
            ax.set_zlabel("$s$")
            ax.set_xlabel("$\\phi_{+}+\\phi_{-}$")
            ax.set_ylabel("$\\phi_{+}-\\phi_{-}$")
            savegraph(fig, basename=basename+"-surf")
    
    return

def main() -> None:
    rrvals = np.array([0.80, 0.85, 0.9, 0.95])
#    graph_00(rrvals, plotsurface=False, basename="graph-00")
    graph_01(rrvals, plotsurface=False, basename="graph-01")
#    graph_02(rrvals, basename="graph-02")
#    graph_03(rrvals, plotsurface=False, basename="graph-03")
#    graph_04(np.array([0.80]), plotsurface=False, basename="graph-04")
    for rr in rrvals:
        phi = 0.0
        zetamax, smax = szetamax(phi, rr)
        zetath = zetathresh(phi, rr)
        print("R=%1.2f: zeta_th = %1.4f, zeta_max/zetath = %1.4f, s_max = %1.4f"
              %(rr, zetath, zetamax/zetath, smax))
    # implicittest()
    return

if __name__ == "__main__":
    main()
