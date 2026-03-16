#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Routine saveStokesParameters for saving supplied Stokes parameters to a CSV
or Poincare-formatted [1] file. This file is typically formatted in such a
way that it is directly compatible as input to the Poincare program [1] for
mapping Stokes parameters as trajectories onto the Poincaré sphere, for
convenient interpretation of the evolution of a polarization state.

Example of usage

    from savestokes import saveStokesParameters
    saveStokesParameters(zn, s0_pump, s1_pump, s2_pump, s3_pump,
                "stokes-pump", fmt="%1.7f", delimiter=" ",
                saveformat="poincare", tickspacing=0.05,
                ticklabel=False, ticklabelspacing=0.2)

References

   [1] "Mapping of Stokes parameter trajectories onto the Poincaré sphere";
       C-code for reading Stokes parameter triplets from an external file,
       of the Poincare format, generating MetaPost code describing the
       mapping of the read trajectory or trajectories of Stokes parameters
       onto the Poincaré sphere. Open source GitHub repo located at
       https://github.com/hp35/poincare

Parameters
----------
savestokes : str
    The base name to use for the filename to which the Stokes
    parameters are to be saved. The resulting filename will be
    <savestokes>-stokes-<[idler|signal|pump]>-<zeta-coordinate>.csv
fmt : string, optional
    Floating point formatting of data. The default is "%f".
delimiter : string, optional
    DESCRIPTION. The default is " ".
header : string, optional
    Example: "header":"p b urgt \"LCP\"". The default is "p".
footer : string, optional
    Example: "footer":"q e lrgt \"RCP\"". The default is "q".
saveformat : str, optional
    The format of the saved Stokes parameters. Use saveformat="csv" in
    order to save the entire set as a regular array in a CSV-file, and
    saveformat="poincare" in order to generate a format for direct
    input to the Poincare (https://hp35/poincare/) visualization
    program.
tickspacing : float, optional
    The spacing between tick marks added to the trajectories,
    applicable only to the saveformat="poincare" option.
ticklabel : bool, optional
    The default is False.
ticklabelspacing : float, optional
    The optional spacing between tick labels (in text) to be added along
    the trajectories, applicable only to the saveformat="poincare" option.

Returns
-------
None.
"""
import datetime
import numpy as np

def saveStokesParameters(zn, s0, s1, s2, s3, savestokes,
                         fmt="%f", delimiter=" ", header="p", footer="q",
                         saveformat="poincare", tickspacing=0.05,
                         ticklabel=False, ticklabelspacing=0.2):
    stokesparams = np.vstack((s1,s2,s3)).transpose()
    if saveformat == "csv":

        """ Save Stokes parameters to file in CSV format """
        sfilename = "%s.csv"%(savestokes)
        kwargs={"fmt":fmt, "delimiter":delimiter, "comments":"",
                "header":header, "footer":footer}
        print("Saving Stokes parameters to %s"%(sfilename))
        np.savetxt(sfilename, stokesparams, **kwargs)

    elif saveformat == "poincare":

        """ Save Stokes parameters to file in Poincare format """
        now = datetime.datetime.now()
        sfilename = "%s.dat"%(savestokes)
        with open(sfilename, "w") as f:
            f.write("p %% %s\n"%(now))
            print("Saving Stokes parameters to %s"%(sfilename))
            dz=zn[1]-zn[0]
            for k in range(len(zn)):
                s1 = stokesparams[k,0]
                s2 = stokesparams[k,1]
                s3 = stokesparams[k,2]
                f.write("%1.6f %1.6f %1.6f"%(s1,s2,s3))
#                if labelzeta:
#                    if (abs(si[k]-self.ds/2) < 1.0e-3):
#                        f.write(" t l lft \"$\\zeta=%1.2f$\"\n"%(self.zcurr))
                if (abs((zn[k]%tickspacing)-dz/2) < 1.0e-3):
                    if ticklabel:
                        if (abs((zn[k]%ticklabelspacing)-dz/2) < 1.0e-3):
                            f.write(" t l rgt \"$%1.2f$\"\n"%(zn[k]-dz))
                        else:
                            f.write(" t %% \"$s=%1.4f$\"\n"%(zn[k]-dz))
                    else:
                        f.write(" t %% \"$s=%1.4f$\"\n"%(zn[k]-dz))
                else:
                    f.write(" %% \"$s=%1.4f$\"\n"%(zn[k]-dz))
            f.write("q\n")
            f.close()

    return
