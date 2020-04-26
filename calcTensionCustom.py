# Assuming Salt Free SDS system
# my netcdf files got corrupted... have to jerry-rig to use pdb files instead (10x less data, ~180 frames)
# note: pdb is reported to only 5~6 sig figs! i.e. limited resolution for doing test area rescaling,
# where roundoff is same order as small perturbations

# System stuff
from sys import stdout
import sys,getopt
import time
import numpy as np
import argparse as ap

"""
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel('INFO')
sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(fmt='%(asctime)s - %(message)s', datefmt="%H:%M:%S")
sh.setFormatter(formatter)
"""
# OpenMM Imports
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit
# ParmEd & MDTraj Imports
from parmed import gromacs
gromacs.GROMACS_TOPDIR = "/home/kshen/mylib/ff"
from parmed.openmm.reporters import NetCDFReporter
from parmed import unit as u
import parmed as pmd
import mdtraj
from pymbar import timeseries
#import MDAnalysis as mda

# Command line inputs
ewldTol = 1e-6

#=== parse: ===
parser = ap.ArgumentParser(description="Calculate surface tension using perturbation. Not suitable for square well potential.")
parser.add_argument('-top', type=str, help="gromacs topology file")
parser.add_argument('-traj', type=str, help="trajectory file, expect includes box info")
parser.add_argument('-box', type=str, help="box file")
parser.add_argument('-pdb', type=str, help="pdb file to read in trajectory")
parser.add_argument('-outprefix', default='', type=str, help="prefix for output files")
parser.add_argument('-dAfrac', type=float, help="fractional perturbation of area")
parser.add_argument('-ewldTol', default=1e-6, type=float, help="ewld tolerance. default is 1e-6")
parser.add_argument('-axis', default=2, choices=[0,1,2], type=int, help="long axis, 0:x, 1:y, 2:z")
parser.add_argument('-stride',default=1, type=int, help="stride for reading trajectory")
parser.add_argument('-volumeChange', action='store_true', help="whether or not the box size changes")
parser.add_argument('-customff',default='', type=str, help='customff file')
parser.add_argument('-deviceid',default=-1,type=int,help="gpu device id")
parser.add_argument('-LJPME',action='store_true', help="default is just PME, toggle for LJPME")
args = parser.parse_args()

ewldTol = args.ewldTol
dAfrac = args.dAfrac
axis = args.axis
print(sys.argv)
print(args)

# files
coord_file  = args.traj # 'output.dcd'
top_file = args.top # gromacs topology file
box_file = args.box # 'box.gro', in future read from trajectory
pdb_file = args.pdb # pdb file to read in trajectory
defines = {}
customff = args.customff

# Test area parameters
#dAfrac   = 0.00001
#scaling1 = expanding area, scaling2 = shrinking area
scaleZ1     = 1.0/(1.0 + dAfrac)
scaleXY1    = (1.0 + dAfrac)**0.5
scaleZ2     = 1.0/(1.0 - dAfrac)
scaleXY2    = (1.0 - dAfrac)**0.5

# === Parameters, assumed, eventually should read in from mdparse ===
T               = 298.
NPT             = False
useLJPME        = args.LJPME #True  #True
LJcut           = 12    #Angstroms
tail            = False#False
Temp            = T     #K
Pressure        = 1.    #bar
barostatfreq    = 25    #time steps
fric            = 1.0   #1/ps
dt              = 2.0   #fs
rigidH2O        = True

# === Start making system ===
gro = gromacs.GromacsGroFile.parse(box_file)
pdb = pmd.load_file(pdb_file)

box = gro.box #note stored in Angstrom
if axis == 2:
    scaling1 = [scaleXY1, scaleXY1, scaleZ1, 1.0, 1.0, 1.0]
    scaling2 = [scaleXY2, scaleXY2, scaleZ2, 1.0, 1.0, 1.0]
elif axis == 1:
    scaling1 = [scaleXY1, scaleZ1, scaleXY1, 1.0, 1.0, 1.0]
    scaling2 = [scaleXY2, scaleZ2, scaleXY2, 1.0, 1.0, 1.0]
elif axis == 0:
    scaling1 = [scaleZ1, scaleXY1, scaleXY1, 1.0, 1.0, 1.0]
    scaling2 = [scaleZ2, scaleXY2, scaleXY2, 1.0, 1.0, 1.0]
box1 = gro.box*scaling1
box2 = gro.box*scaling2


top = gromacs.GromacsTopologyFile(top_file, defines=defines, box=box)
top1= gromacs.GromacsTopologyFile(top_file, defines=defines, box=box1)
top2= gromacs.GromacsTopologyFile(top_file, defines=defines, box=box2)
#top.positions = pdb.positions


def makeSystem(top):#, Temp, useLJPME, LJcut, tail=True, NPT=False, Pressure=1, barostatfreq=25, rigidH2O=True)
    if useLJPME:
        nbm = mm.NonbondedForce.LJPME
    else:
        nbm = mm.NonbondedForce.PME

    system = top.createSystem(nonbondedMethod=app.PME,
                nonbondedCutoff=LJcut*u.angstroms,
                constraints=app.HBonds, rigidWater=rigidH2O, ewaldErrorTolerance=ewldTol) #default ewTol=5e-4

    ftmp = [f for ii, f in enumerate(system.getForces()) if isinstance(f,mm.NonbondedForce)]
    fnb = ftmp[0]
    fnb.setNonbondedMethod(nbm)
    print("Nonbonded method, use: {}".format(fnb.getNonbondedMethod()) )

    if (not tail) or (useLJPME):
        print("Turning off tail correction...")
        fnb.setUseDispersionCorrection(False)
        print("Check dispersion flag: {}".format(fnb.getUseDispersionCorrection()) )

    if NPT:
        barostat = mm.MonteCarloBarostat(Pressure*u.bar, Temp*u.kelvin, barostatfreq)
        system.addForce(barostat)
         
    integrator = mm.LangevinIntegrator(
		    Temp*u.kelvin,       # temperature of heat bath
		    fric/u.picoseconds,  # friction coefficient
		    dt*u.femtoseconds, # time step
    )

    if customff:
        #logger.info("Using customff: [{}]".format(customff))
        print("Using customff: [{}]".format(customff))
        with open(customff,'r') as f:
            ffcode = f.read()
        exec(ffcode,globals(),locals()) #python 3, need to pass in globals to allow exec to modify them (i.e. the system object)
    else:
        #logger.info("--- No custom ff code provided ---")
        print("--- No custom ff code provided ---")
    for f in system.getForces():
        print('...force {}'.format(f))
 

    return system,integrator

system0, integrator0 = makeSystem(top)#, Temp=Temp, useLJPME=useLJPME, LJcut=LJcut, tail=tail, NPT=NPT, Pressure=Pressure, barostatfreq=barostatfreq, rigidH2O=rigidH2O)
system1, integrator1 = makeSystem(top1)
system2, integrator2 = makeSystem(top2)
properties = {'OpenCLPrecision': 'double'}
platform = mm.Platform.getPlatformByName('OpenCL')
platform.setPropertyDefaultValue('Precision','double')
if args.deviceid >= 0:
    print("setting device to {}".format(args.deviceid))
    platform.setPropertyDefaultValue('OpenCLDeviceIndex',str(args.deviceid))
sim0 = app.Simulation(top.topology,  system0, integrator0, platform)
sim1 = app.Simulation(top1.topology, system1, integrator1, platform)
sim2 = app.Simulation(top2.topology, system2, integrator2, platform)
#sim0 = app.Simulation(top.topology,  system0, integrator0, platformProperties=properties)
#sim1 = app.Simulation(top1.topology, system1, integrator1, platformProperties=properties)
#sim2 = app.Simulation(top2.topology, system2, integrator2, platformProperties=properties)

#simulation.context.setPositions(pdb.positions)
#simulation.context.applyConstraints(1e-12)


# === CALCULATE ENERGIES ===
def getEnergy(simulation,positions):
    simulation.context.setPositions(positions)
    state = simulation.context.getState( getEnergy=True )
    return state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)

traj = mdtraj.load(coord_file,top=pdb_file, stride=args.stride)
energies = np.zeros([traj.n_frames, 3])

print("===== starting energy calculations =====")
start = time.time()
for iframe,ts in enumerate(traj):
    if( np.mod(iframe,10)==0 or iframe == traj.n_frames-1 ):
        print("On frame {}".format(iframe))
        np.savetxt('{}_energies.txt'.format(args.outprefix),energies,delimiter=',', header="dA% = {} area change".format(dAfrac))
        np.save('{}_energies'.format(args.outprefix),energies)
    
    if args.volumeChange:
        box = ts.openmm_boxes(0)
        print("Changing box... {}".format(box))
        sim0.context.setPeriodicBoxVectors(box[0], box[1], box[2])
        sim1.context.setPeriodicBoxVectors(box[0]*scaling1[0], box[1]*scaling1[1], box[2]*scaling1[2])
        sim2.context.setPeriodicBoxVectors(box[0]*scaling2[0], box[1]*scaling2[1], box[2]*scaling2[2])
        print(sim2.context.getState().getPeriodicBoxVectors())

    pos = ts.xyz[0] #pdb in A, but openMM default seems to be in nm
    en0 = getEnergy(sim0,pos)

    natoms = len(top.atoms)
    reference = np.zeros([natoms,3])
    for res in top.residues:
        atomids = [atom.idx for atom in res.atoms]
        com = np.mean( pos[atomids,:], 0 )
        reference[atomids,:] = com[None,:]


    #calculate larger box
    if axis == 2:
        newpos = pos + reference*np.array( [scaleXY1-1.0, scaleXY1-1.0, scaleZ1-1.0] )    
    elif axis == 1:
        newpos = pos + reference*np.array( [scaleXY1-1.0, scaleZ1-1.0, scaleXY1-1.0] )    
    elif axis == 0:
        newpos = pos + reference*np.array( [scaleZ1-1.0, scaleXY1-1.0, scaleXY1-1.0] )    
    en1    = getEnergy(sim1,newpos)    

    #calculate smaller box
    if axis == 2:
        newpos = pos + reference*np.array( [scaleXY2-1.0, scaleXY2-1.0, scaleZ2-1.0] )
    elif axis == 1:
        newpos = pos + reference*np.array( [scaleXY2-1.0, scaleZ2-1.0, scaleXY2-1.0] )
    elif axis == 0:
        newpos = pos + reference*np.array( [scaleZ2-1.0, scaleXY2-1.0, scaleXY2-1.0] )
    en2    = getEnergy(sim2, newpos)

    energies[iframe,:] = np.array([en0, en1, en2])

end = time.time()
print("Took {} seconds".format(end-start))

#np.savetxt('energies.txt',energies,delimiter=',',header="dA% = {} area change; Col0: Reference, Col1: +dA, Col2: -dA".format(dAfrac))
np.savetxt('{}_energies.txt'.format(args.outprefix),energies,delimiter=',', header="dA% = {} area change".format(dAfrac))
np.save('{}_energies'.format(args.outprefix),energies)

# === detect correlations in energies ===
[nequil,g,Neff_max] = timeseries.detectEquilibration(energies[:,0])
indices = timeseries.subsampleCorrelatedData(energies[:,0], g=g)


# === ESTIMATE SURFACE TENSION ===
# finite differencing with area change 2dA, account for two interfaces
kb  = 1.3806504e-23     #1.38064852e-23    #J/K
kT  = kb*Temp           #J
Nav = 6.02214179e23     #6.022e23
kTkJmol = kT*Nav/1000   #kT in kJ/mol

dE1 = energies[indices,1] - energies[indices,0]
dE2 = energies[indices,2] - energies[indices,0]

# --- Using exponential averaging, ok only for small energy changes ---
BdE1 = dE1/kTkJmol
BdE2 = dE2/kTkJmol

dF1 = -kT * np.log( np.exp(-BdE1).mean() ) #J
dF2 = -kT * np.log( np.exp(-BdE2).mean() ) #J

dF1error = kT/np.abs(np.exp(-BdE1).mean())*np.exp(-BdE1).std()/np.sqrt(np.size(BdE1))
dF2error = kT/np.abs(np.exp(-BdE2).mean())*np.exp(-BdE2).std()/np.sqrt(np.size(BdE2))


if axis == 2:
    da = 2 * dAfrac * (gro.box[0]/10.0) * (gro.box[1]/10.0)  # nm
elif axis == 1:
    da = 2 * dAfrac * (gro.box[0]/10.0) * (gro.box[2]/10.0)  # nm
elif axis == 0:
    da = 2 * dAfrac * (gro.box[1]/10.0) * (gro.box[2]/10.0)  # nm

tension = (dF1 - dF2)/2/da * 1e18 #convert from J/nm^2 to  J/m^2 aka N/m
tensionError = np.sqrt(dF1error**2 + dF2error**2)/2/da*1e18
print('tension: {} +/- {}N/m'.format(tension, tensionError)) 

np.savetxt('energies.txt',energies,delimiter=',',
        header="dA% = {} area change, gamma={} N/m; Col0: Reference, Col1: +dA, Col2: -dA".format(dAfrac,tension))

import datetime
with open('results.txt',"a") as f:
    f.write('{}\n'.format(datetime.datetime.now()) )
    f.write('{}\n'.format(sys.argv))
    f.write('T={}K, dA % = {} area change, gamma = {}+/-{} N/m\n'.format(Temp,dAfrac, tension, tensionError)),
    f.write('dFfwd = {}+/-{}J/mol, dFback={}+/-{}J/mol\n'.format(dF1, dF1error, dF2, dF2error))
    f.write('or dFfwd = {}+/-{}kT, dFback = {}+/-{}kT\n'.format(dF1/kT, dF1error/kT, dF2/kT, dF2error/kT))


# --- using pymbar to do Bennet's, assuming A+dA --> A is the same as A --> A-dA  --- #
#     in contrast, above we were doing A+dA <--> A-dA, so there was a 2dA area change
from pymbar import MBAR, timeseries
print("=== Now trying pymbar ===")
nstates = 2
nframes = len(dE1)
u_kln = np.zeros([nstates,nstates,nframes], np.float64)
u_kln[0,1,:] = BdE1
u_kln[1,0,:] = BdE2

# get uncorrelated samples
print("=== Getting uncorrelated samples===")
N_k = np.zeros([nstates], np.int32) # number of uncorrelated samples
for k in range(nstates):
    [nequil, g, Neff_max] = timeseries.detectEquilibration(u_kln[k,k,:])
    indices = timeseries.subsampleCorrelatedData(u_kln[k,k,:], g=g)
    N_k[k] = len(indices)
    u_kln[k,:,0:N_k[k]] = u_kln[k,:,indices].T
print("...found {} uncorrelated samples...".format(N_k))

np.save('{}_ukln'.format(args.outprefix),u_kln)

# Compute free energy differences and statistical uncertainties
print("=== Computing free energy differences ===")
mbar = MBAR(u_kln, N_k)
[DeltaF_ij, dDeltaF_ij, Theta_ij] = mbar.getFreeEnergyDifferences()

np.savetxt('{}_DeltaF.dat'.format(args.outprefix),DeltaF_ij)
np.savetxt('{}_dDeltaF.dat'.format(args.outprefix),dDeltaF_ij)

# Print out one line summary 
#tension = DeltaF_ij[0,1]/2/da * 1e18
#tensionError = dDeltaF_ij[0,1]/2/da * 1e18
tension = DeltaF_ij[0,1]/da * 1e18 * kT #(in J/m^2). note da already has a factor of two for the two areas!
tensionError = dDeltaF_ij[0,1]/da * 1e18 * kT
print('tension (pymbar): {} +/- {}N/m'.format(tension,tensionError))

with open('{}_results.txt'.format(args.outprefix),"a") as f:
    f.write('\nUsing pymbar:\n')
    f.write('dF = {} +/- {}kT\n'.format(DeltaF_ij[0,1],dDeltaF_ij[0,1]))
    f.write('tension = {} +/- {} N/m'.format(tension, tensionError))




