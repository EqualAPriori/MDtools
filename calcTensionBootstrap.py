# Takes in previously calculated energy data (A, A+dA, A-dA)
# and cross-validates by subdividing into blocks and recalculating free energy
#
import numpy as np
import scipy
from pymbar import MBAR, timeseries

from parmed import gromacs
gromacs.GROMACS_TOPDIR = "/home/kshen/mylib/ff"

import argparse as ap
parser = ap.ArgumentParser(description="Boot-strap energy data to estimate error")
parser.add_argument('energy_file',type=str,help='energy file, 3 columns, A, A+dA, A-dA')
parser.add_argument('-dAfrac',type=float,default=1e-4,help='fractional perturbation of area')
parser.add_argument('-nboot',type=int,default=100,help='# bootstrap trials')
parser.add_argument('-verbose',action='store_true',help='toggle on verbose output')
args = parser.parse_args()

dAfrac = args.dAfrac
box_file = 'box.gro'
gro = gromacs.GromacsGroFile.parse(box_file)
da = 2 * dAfrac * (gro.box[0]/10.0) * (gro.box[1]/10.0)


Temp = 298
kb  = 1.3806504e-23     #1.38064852e-23    #J/K
kT  = kb*Temp           #J
Nav = 6.02214179e23     #6.022e23
kTkJmol = kT*Nav/1000   #kT in kJ/mol
niters = args.nboot

datafile = args.energy_file#'energies.txt'
if args.energy_file.endswith('npy'):
    energies = np.load(args.energy_file)
else:
    try:
        energies =  np.loadtxt(datafile)
    except:
        energies =  np.loadtxt(datafile,delimiter=',')


def calcTension(energy_data,verbose=False):
    dE1 = energy_data[:,1] - energy_data[:,0]
    dE2 = energy_data[:,2] - energy_data[:,0]
    BdE1 = dE1/kTkJmol
    BdE2 = dE2/kTkJmol

    nstates = 2
    nframes = len(dE1)
    u_kln = np.zeros([nstates,nstates,nframes], np.float64)
    u_kln[0,1,:] = BdE1
    u_kln[1,0,:] = BdE2

    N_k = np.zeros([nstates], np.int32) # number of uncorrelated samples
    for k in range(nstates):
        [nequil, g, Neff_max] = timeseries.detectEquilibration(u_kln[k,k,:])
        indices = timeseries.subsampleCorrelatedData(u_kln[k,k,:], g=g)
        N_k[k] = len(indices)
        u_kln[k,:,0:N_k[k]] = u_kln[k,:,indices].T
    if verbose: print("...found {} uncorrelated samples out of {} total samples...".format(N_k,nframes))

    if verbose: print("=== Computing free energy differences ===")
    mbar = MBAR(u_kln, N_k)
    [DeltaF_ij, dDeltaF_ij, Theta_ij] = mbar.getFreeEnergyDifferences()

    tension = DeltaF_ij[0,1]/da * 1e18 * kT #(in J/m^2). note da already has a factor of two for the two areas!
    tensionError = dDeltaF_ij[0,1]/da * 1e18 * kT
    if verbose: print('tension (pymbar): {} +/- {}N/m'.format(tension,tensionError))

    return tension, tensionError

nsamples = energies.shape[0]
results = np.zeros(niters)
for ii in range(niters):
    indices = np.random.choice(nsamples,size = energies.shape[0], replace=True)
    if np.mod(ii,100) == 0:
        tension, tensionError = calcTension( energies[indices,:], True )
    else:
        tension, tensionError = calcTension( energies[indices,:], args.verbose )
    results[ii] = tension

np.savetxt('tension_bootstrap.txt',results,header='bootstrap sample properties: {}N/m, std {}N/m'.format(results.mean(),results.std()))
print(results)
print('bootstrap sample properties: {}N/m, std {}N/m'.format(results.mean(),results.std()))
tension,tensionError = calcTension(energies)
print('compare to using full sample data: {}N/m +/- {}N/m'.format(tension,tensionError))

