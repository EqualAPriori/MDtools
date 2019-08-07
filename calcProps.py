#!/usr/b.in/env python

# (C) Kevin Shen, initialized 2019.06.17
# For calculation of properties of molecular systems
#

import numpy as np
import mdtraj as md


def epsr( traj, qs, T=298.0 ):
    """ Read and strips the water from a trajectory file. Uses MDtraj

    Parameters
    ----------
    traj : array
        trajectory object mdtraj format (nframes, natoms, xyz). Assumes units of nanometers.  
    qs : array
        vector of charges of the atoms/particles in each frame
    T : float
        temperature. default is 298K.

    Returns
    -------
    eps 
        double, relative dielectric permittivity

    Notes
    -----
    Currently does not calculate periodic images (tested on a 3nm water box, only led to difference on 10th decimal point)
    Calculates epsr = 1 + 4*pi*lb0 <M^2> / 3V
    Shifts box by [Lx,Ly,Lz]/2.0, i.e. to center an openMM trajectory. Although in tests shouldn't really significantly change properties (changes in 7th decimal point)
    If NPT box, divides by the volume mean... unclear if that's the appropriate thing to do, or if should divide M2 by volume before averaging.

    References
    ----------
    [1] http://mdtraj.org/1.9.0/
    """

    nmol = traj.n_residues

    box = traj.unitcell_lengths
    V = traj.unitcell_volumes
    xyz = traj.xyz - box[:,None,:]/2.0

    dip = xyz*qs[None,:,None]
    M = np.sum(dip,1)
    M2 = np.sum(M*M,1)

    Tref = 298.0
    lb0 = 56.0742*Tref/T #nm
    eps = 1 + 4*np.pi*lb0/3.0/V.mean() * M2.mean()
    #epsframe = 1 + 4*np.pi*lb0/3.0/V * M2
    #eps = epsframe.mean()
    #std = epsframe.std()
    std = 4*np.pi*lb0/3.0/V.mean()*M2.std()

    print("<M>: {}, <M2>: {}".format(M.mean(), M2.std()))
    #np.savetxt('eps.dat',(np.vstack([np.arange(xyz.shape[0]),1 + 4*np.pi*lb0/3.0/V.mean() * M2]).T))
    np.savetxt('eps.dat',(np.vstack([np.arange(xyz.shape[0]),1 + 4*np.pi*lb0/3.0/V * M2]).T))

    return eps,std
