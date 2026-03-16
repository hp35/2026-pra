#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 15:10:37 2025

@author: frejon
"""
from scipy.special import ellipj, ellipk
from sympy import symbols, idiff, simplify, sqrt

s, phi, r = symbols('s phi r')

cn = ellipj



s2, phi2 = s*s, phi*phi

f2 = sqrt(np.square(1+s2+phi2)-4.0*phi2)
f = sqrt(f2)


xi = np.divide(1.0-s2-f2+fg2,fg2)/2.0
kk = ellipk(xi)
sn, cn, dn, ph = ellipj(fg*zeta+kk, xi)



phig2 = np.square(phig)
sg2 = np.square(sg)

fig, ax = plt.subplots(figsize=(5.2,4.0))
ccolors, clinestyles, clabels = [], [], []
for k, rr in enumerate(rrvals):
    color, linestyle, labeltext = 'k', linestyles[k], '$R_{\\pm}=%1.2f$'%(rr)
    z = (1-sg2-phig2+fg2)/2.0
    z *= np.square(cn)
    z -= ((1.0-rr)/rr)*sg2


ex = x**5 + y**2 + z**4 - 8*x*y*z
ex_d = simplify(idiff(ex,(x,y),z))