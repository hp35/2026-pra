#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 13:57:27 2025

@author: frejon
"""
    # def f(s, phi, rr):
    #     s2, phi2 = np.square(s), np.square(phi)
    #     f2 = np.sqrt(np.square(1.0 + s2 + phi2) - 4.0*phi2)
    #     u3a = (1.0+s2+phi2-f2)/2.0
    #     u3b = 1.0
    #     u3c = (1.0+s2+phi2+f2)/2.0
    #     xi2 = np.divide(u3b-u3a, u3c-u3a)
    #     # xi2 = (1.0+np.divide(1.0-s2-phi2,f2))/2.0
    #     cc = 2.0*(1.0-rr)/rr
    #     return cc*np.multiply(s2,xi2) + np.multiply(1.0-s2-phi2+f2,1.0-xi2)


    # def f(s, phi, rr):
    #     s2, phi2 = np.square(s), np.square(phi)
    #     f2 = np.sqrt(np.square(1.0 + s2 + phi2) - 4.0*phi2)
    #     u3a = (1.0+s2+phi2-f2)/2.0
    #     u3b = 1.0
    #     u3c = (1.0+s2+phi2+f2)/2.0
    #     xi2 = np.divide(u3b-u3a, u3c-u3a)
    #     # xi2 = (1.0+np.divide(1.0-s2-phi2,f2))/2.0
    #     cc = 2.0*(1.0-rr)/rr
    #     return cc*np.multiply(s2,xi2) + np.multiply(1.0-s2-phi2+f2,1.0-xi2)

    # func = lambda s : f(s, phi, rr)

    # if plotf:
    #     s = np.linspace(0.0,5.0,1000)
    #     fig, ax = plt.subplots(figsize=(5.2,4.0))
    #     ax.plot(s, f(s,phi,rr), color='#1f77b4ff', linewidth=1.5)
    #     ax.set_xlabel('$s$')
    #     ax.set_ylabel('$f(s,\\phi,R)$')
    #     ax.set_title('Function $f(s,\\phi,R)$ used for optimization '
    #               '($\\phi=$%1.2f, $R=%1.2f$)'%(phi,rr))
    #     ax.grid(visible=True, which='major', axis='both', color=gridcolor)
    #     ax.tick_params(which="both", top=True, right=True, labeltop=False, 
    #                    bottom=True, labelbottom=True, direction="in")

    # if plotcntr:
    #     zetanormmax = 3.0
    #     smax = 3.0
    #     phi = 0.0
    #     phi2 = np.square(phi)
    #     zetanorm = np.linspace(-zetanormmax/4, zetanormmax, 1024)
    #     s = np.linspace(-smax/4, smax, 1024)
    #     zetanorm2 = np.square(zetanorm)
    #     s2 = np.square(s)
    #     zetanormg2, sg2 = np.meshgrid(zetanorm2,s2)
    #     fg2 = np.sqrt(np.square(1.0 + sg2 + phi2) - 4.0*phi2)
    #     fg = np.sqrt(fg2)
    #     u3a = (1.0+s2+phi2-fg2)/2.0
    #     u3b = 1.0
    #     u3c = (1.0+s2+phi2+fg2)/2.0
    #     xi2 = np.divide(u3b-u3a, u3c-u3a)
    #     xi = np.sqrt(xi2)
    #     cc = 2.0*(1.0-rr)/rr
    #     m = xi2
    #     sn, cn, dn, ph = ellipj(fg*np.sqrt(zetanormg2), m)
    #     z = np.multiply(1.0-sg2-phi2+fg2,1.0-xi2)
    #     z += cc*np.multiply(xi2,sg2)
    #     z = np.multiply(z,np.square(sn))
    #     z -= cc*sg2

    #     """
    #     Plot the surface $G(\zeta,s)$, which at $G(\zeta,s)=0$ corresponds
    #     to the function $s(\zeta)$.
    #     """
    #     fig, ax = plt.subplots(figsize=(5.4,5.4),
    #                            subplot_kw={"projection":"3d"})
    #     ax.plot_surface(zetanormg2, sg2, z)
    #     ax.set_xlabel("$(\\zeta/\\zeta^{({\\rm th})})^2$")
    #     ax.set_ylabel("$s^2_{\\pm}$")
    #     ax.set_zlabel("$G(\\zeta,s)$")
    
    #     """
    #     Plot the contour at $G(\zeta,s)=0$, corresponding to the function
    #     $s(\zeta)$.
    #     """
    #     fig, ax = plt.subplots(figsize=(5.4,5.4))
    #     ax.contour(zetanormg2, sg2, z, [0], zdir='z', colors='#1f77b4ff',
    #                linewidths=1.5, linestyles='solid')
    #     ax.grid(visible=True, which='major', axis='both', color=gridcolor)
    #     ax.tick_params(which="both", top=True, right=True, labeltop=False, 
    #                    bottom=True, labelbottom=True, direction="in")
    #     ax.set_xlabel("$(\\zeta/\\zeta^{({\\rm th})})^2$")
    #     ax.set_ylabel("$s^2_{\\pm}$")



    #     fig, ax = plt.subplots(2, figsize=(5.4,5.4))
    #     u = np.linspace(-5.0, 5.0, 1024)
    #     xi = 0.9
    #     xi2 = xi*xi
    #     m = xi2
    #     sn, cn, dn, ph = ellipj(u, m)
    #     kk = ellipk(m)
    #     snk, cnk, dnk, phk = ellipj(u+kk, m)
    #     cnk2 = np.square(cnk)
    #     sn2 = np.square(sn)
    #     cnk2equiv = (1.0-xi2)*np.divide(sn2,(1.0-xi2*sn2))
    #     diff = cnk2-cnk2equiv

    #     ax[0].plot(u, cnk2, color='#1f77b4ff', linewidth=1.5, 
    #                label='${\\rm cn}^2(u+K,\\xi)$')
    #     ax[0].plot(u, cnk2equiv, color='xkcd:tomato', linewidth=1.5, 
    #                label='$\\displaystyle\\frac{(1-\\xi^2)\,
    # {\\rm sn}^2(u,\\xi)}{1-\\xi^2\,{\\rm sn}^2(u,\\xi)}$')
    #     ax[0].grid(visible=True, which='major', axis='both', color=gridcolor)
    #     ax[0].tick_params(which="both", top=True, right=True, labeltop=False, 
    #                    bottom=True, labelbottom=True, direction="in")
    #     ax[0].legend(loc='upper right')
    #     ax[1].plot(u, diff, color='#1f77b4ff', linewidth=1.5, linestyle='solid')
    #     ax[1].grid(visible=True, which='major', axis='both', color=gridcolor)
    #     ax[1].tick_params(which="both", top=True, right=True, labeltop=False, 
    #                    bottom=True, labelbottom=True, direction="in")
    #     ax[1].set_xlabel("$u$")
    #     ax[1].set_ylabel("diff against ${\\rm cn}^2(u+K,\\xi)$")

    # s_ini = 2.0
    # ssol = fsolve(func, s_ini)
    # zetamax = 0.0
    # smax = ssol[0]
    # print("smax = %1.4f"%smax)
