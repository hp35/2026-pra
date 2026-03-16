#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of the quasi-periodicity of the incomplete elliptic integral of
the third kind.

Copyright (C) 2026, Fredrik Jonsson under GPLv3.
File: copaproc/python/periodicity.py, created on Mon Feb 16 08:07:11 2026.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpmath import ellippi    # Incomplete elliptic integral of third kind

"""
As a global standard, use TeX-style labeling for everything graphics-related.
"""
plt.rcParams.update({
    "text.usetex" : True,
    "font.family" : "Computer Modern",
    "font.size"   : 12
})

def main() -> None:
    z = np.linspace(0.0, 10.0, num=200)
    v = np.zeros_like(z, dtype=float)
    alpha = 20.0
    xi = 0.8
    for k in range(len(z)):
        phi = 0.2*z[k]
#        print("k=%d, phi=%1.4f"%(k,phi))
        val = ellippi(alpha*alpha, phi, xi*xi)
        v[k] = float(val.real*val.real+val.imag*val.imag)
#        val = ellippi(alpha*alpha, phi, xi*xi)
#        v[k] = float(np.real_if_close(complex(val)))

    fig, ax = plt.subplots(1,1,figsize=(9.0,6.0))
    major_x_ticks = np.linspace(0.0, 1.0, 6)

    ax.plot(z, v, 'b')
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_ylabel('$\\Pi(\\alpha,\\phi,\\xi)$')
    ax.set_xticks(major_x_ticks)
#    ax.set_yticks(np.linspace(0.0, 1.0, 5))
    ax.grid(which='both')
#    ax.set_title("$\\xi=%1.2f$, $\\alpha=%1.2f$"%(xi, alpha))

    return

if __name__ == "__main__":
    main()#!/usr/bin/env python3
